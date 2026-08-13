"""
批量版（height / intensity 命名规则）+ 两侧边缘噪点抑制（不裁剪工件）。

问题定位
--------
原脚本合成后，工件左右两侧各有一条 250~300 px 宽的彩色/绿色麻点带。实测确认：
  * 该区域 depth==0 的掉点率仅 0.0%，并不是掉点造成的；
  * 而是掠射角处高度值本身抖动大 —— 局部粗糙度 |depth-median5x5| 在两侧约 165，
    工件中部仅约 70，相差 2.3 倍。

真正的放大器是全局 MAD 归一化：尺度由面积占优的安静中部决定，于是两侧较大的
噪声除以偏小的尺度后远远超出 ±1，被 clip 到饱和，在 R/G 通道里就成了麻点。

解决思路（不裁剪，保持棒子原始宽度）
------------------------------------
用**局部噪声尺度**替代全局 MAD：
  1. _local_scale：先算残差的局部离差 |res - blur(res)|，再做掩膜感知大尺度平滑，
     得到逐像素的噪声水平 sigma_local（两侧自然偏大、中部偏小）。
  2. _local_norm：res / (k * max(sigma_local, floor))。同一条真实缺陷在任何位置
     都按当地噪声水平衡量，边缘噪声被就地压回 ±1 以内，不再饱和；
     floor 取全局尺度的一个比例，防止极安静区域把微小噪声放大成假缺陷。
  3. _denoise_by_noise：按 sigma_local 与中部水平的比值做**空间自适应去噪**，
     只在高噪声区（两侧）用中值/高斯结果替换，权重随比值平滑上升，中部完全不动。
  4. 边界渐隐 GUARD_PX（6 px）只压掉形态学/LoG 在轮廓处的假响应，不影响内部。

掩膜只做「补孔 + 去碎点 + 最大连通域 + ERODE_PX 轻微内缩」，不做任何列裁剪，
因此工件宽度与原脚本一致，不会把棒子削细。

本版特点：只输出最终合成图；除输入/输出路径外所有参数按调优后的默认值写死。

用法：
    python synthesize-fast-多图-西克v2.py -i ./imgs -o ./results
"""

import os
import argparse
import warnings
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================================== #
#                    固定参数（调优后写死，不再对外开放）
# =========================================================================== #
SIGMA_BASELINE = 60.0     # 低频基线高斯 sigma
ERODE_PX = 8              # 有效区内缩像素（与原脚本一致，不改变棒子宽度）
GUARD_PX = 6              # 轮廓渐隐宽度，仅压边界假响应
K_LOCAL = 6.0             # 局部归一化系数
NR_STRENGTH = 1.5         # 高噪声区自适应去噪强度
SCALE_SIGMA = 48.0        # 噪声尺度场平滑 sigma
FILL_SIGMA = 24.0         # 掉点填充用高斯 sigma
FINE_SIGMA = 2.0          # 噪声尺度估计的高频提取 sigma
FLOOR_FRAC = 0.6          # 局部尺度下限 = FLOOR_FRAC * 全局 MAD
RATIO_LO = 1.25           # 去噪权重起始噪声比值
RATIO_HI = 2.6            # 去噪权重饱和噪声比值
MED_KSIZE = 5             # 去噪中值核
BLUR_SIGMA = 1.6          # 去噪轻高斯 sigma
OPEN_PX = 3               # 掩膜去碎点
CLOSE_PX = 7              # 掩膜补小孔
RECURSIVE = True          # 递归扫描子目录

TH_RADII = (5, 11, 21)          # 缺陷通道 Top-Hat 多尺度半径
LOG_SIGMAS = (1.2, 2.0, 3.0)    # 缺陷通道 LoG 多尺度 sigma
ALPHA_TH = 0.6                  # Top-Hat 融合权重
ALPHA_LOG = 0.4                 # LoG 融合权重
CLAHE_CLIP = 3.0                # 亮度通道 CLAHE
CLAHE_TILE = 16


# =========================================================================== #
#                                 输入清洗
# =========================================================================== #
def _ensure_depth16(d: np.ndarray) -> np.ndarray:
    if d.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {d.shape}")
    if d.dtype == np.uint16:
        return d
    if d.dtype == np.uint8:
        return d.astype(np.uint16)
    return np.clip(d, 0, 65535).astype(np.uint16)


def _ensure_intensity8(i: np.ndarray) -> np.ndarray:
    if i.ndim != 2:
        raise ValueError(f"intensity must be 2D, got shape {i.shape}")
    if i.dtype == np.uint8:
        return i
    if i.dtype == np.uint16:
        return (i >> 8).astype(np.uint8)
    return np.clip(i, 0, 255).astype(np.uint8)


# =========================================================================== #
#                                 基础工具
# =========================================================================== #
def _mask_aware_gaussian(img: np.ndarray, mask: np.ndarray,
                         sigma: float) -> np.ndarray:
    """掩膜感知高斯：blur(img*mask)/blur(mask)；sigma>=16 时下采样加速"""
    img_f = img.astype(np.float32) * mask.astype(np.float32)
    m_f = mask.astype(np.float32)

    if sigma >= 16.0:
        scale = max(1, int(round(sigma / 8.0)))
        H, W = img.shape
        ds = (max(1, W // scale), max(1, H // scale))
        img_s = cv2.resize(img_f, ds, interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(m_f, ds, interpolation=cv2.INTER_AREA)
        n_s = cv2.GaussianBlur(img_s, (0, 0), sigma / scale)
        d_s = cv2.GaussianBlur(m_s, (0, 0), sigma / scale) + 1e-6
        out_s = n_s / d_s
        return cv2.resize(out_s, (W, H), interpolation=cv2.INTER_LINEAR)

    n = cv2.GaussianBlur(img_f, (0, 0), sigma)
    d = cv2.GaussianBlur(m_f, (0, 0), sigma) + 1e-6
    return n / d


def _remove_row_bias(residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """每行减去其内部中位数（向量化）"""
    masked = np.where(mask > 0, residual, np.nan).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_med = np.nanmedian(masked, axis=1, keepdims=True)
    row_med = np.nan_to_num(row_med, nan=0.0)
    row_count = (mask > 0).sum(axis=1, keepdims=True)
    row_med = np.where(row_count > 32, row_med, 0).astype(np.float32)
    return residual - row_med


# =========================================================================== #
#              核心：局部噪声尺度 / 自适应去噪 / 局部归一化
# =========================================================================== #
def _local_scale(res: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    逐像素噪声尺度 sigma_local。

    |res - blur(res, FINE_SIGMA)| 得到高频幅度，再用掩膜感知大尺度平滑
    (SCALE_SIGMA) 得到平缓的噪声水平场。FINE_SIGMA 小 -> 只看高频噪声，
    不会把真实缺陷的低频形状算进噪声里；SCALE_SIGMA 大 -> 尺度场足够平滑，
    不会被单个缺陷局部抬高（否则缺陷会自己把自己压掉）。
    """
    dev = np.abs(res - _mask_aware_gaussian(res, mask, FINE_SIGMA))
    scale = _mask_aware_gaussian(dev, mask, SCALE_SIGMA)
    return np.maximum(scale, 1e-6).astype(np.float32)


def _denoise_by_noise(res: np.ndarray, mask: np.ndarray,
                      sigma_local: np.ndarray) -> np.ndarray:
    """
    空间自适应去噪：只在噪声比值高的地方（两侧）平滑，中部保持原样。

    w = clip((ratio - LO) / (HI - LO), 0, 1) * NR_STRENGTH
    out = (1 - w) * res + w * smooth
    smooth 用中值(去脉冲) + 轻高斯(去颗粒) 串联，对麻点最有效。
    w 由噪声水平驱动而非位置驱动，所以不会误伤中部的真实缺陷。
    """
    if NR_STRENGTH <= 0:
        return res

    v = sigma_local[mask > 0]
    quiet = max(float(np.percentile(v, 35)), 1e-6) if v.size else 1.0
    ratio = sigma_local / quiet

    w = np.clip((ratio - RATIO_LO) / max(1e-6, RATIO_HI - RATIO_LO), 0.0, 1.0)
    w = np.clip(w * NR_STRENGTH, 0.0, 1.0).astype(np.float32)

    smooth = cv2.medianBlur(res.astype(np.float32), int(MED_KSIZE) | 1)
    if BLUR_SIGMA > 0:
        smooth = cv2.GaussianBlur(smooth, (0, 0), BLUR_SIGMA)

    return ((1.0 - w) * res + w * smooth).astype(np.float32)


def _local_norm(x: np.ndarray, mask: np.ndarray,
                sigma_local: np.ndarray) -> np.ndarray:
    """
    局部归一化到 [-1,1]：x / (K_LOCAL * max(sigma_local, floor))。

    floor = FLOOR_FRAC * 全局 MAD，双向保护：
      * 防止过于安静的区域把微小起伏放大成假缺陷；
      * 保证同尺寸真实缺陷在中部依然有足够对比度。
    """
    v = x[mask > 0]
    if v.size == 0:
        return np.zeros_like(x)
    med = float(np.median(v))
    gmad = float(np.median(np.abs(v - med)))
    if gmad < 1e-3:
        gmad = max(1e-3, 0.05 * float(np.std(v)))

    denom = np.maximum(sigma_local, FLOOR_FRAC * gmad) * K_LOCAL
    return np.clip((x - med) / np.maximum(denom, 1e-6), -1.0, 1.0)


# =========================================================================== #
#              掩膜：补孔 / 去碎点 / 最大连通域 / 轻微内缩
# =========================================================================== #
def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    """只保留面积最大的连通域，丢掉背景里飞溅的孤立噪点块"""
    if not mask_u8.any():
        return mask_u8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


def build_valid_mask(depth16: np.ndarray):
    """
    返回 (raw_mask, mask)。

    只比原脚本多了「补孔 + 去碎点 + 最大连通域」，用来清掉背景中零散的伪点；
    ERODE_PX 仍为 8，因此工件宽度与原脚本一致，不会削细棒子。
    """
    raw_mask = (depth16 > 0).astype(np.uint8)

    solid = raw_mask
    if CLOSE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLOSE_PX + 1,) * 2)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, k)
    if OPEN_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OPEN_PX + 1,) * 2)
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, k)
    solid = _largest_component(solid)

    mask = solid
    if ERODE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * ERODE_PX + 1,) * 2)
        mask = cv2.erode(mask, k, iterations=1)
    if not mask.any():
        mask = solid
    return raw_mask, mask


def _guard_weight(mask: np.ndarray) -> np.ndarray:
    """距掩膜边界 GUARD_PX 内线性渐隐，只压轮廓处的形态学/LoG 假响应"""
    if GUARD_PX <= 0:
        return mask.astype(np.float32)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return np.clip(dist / float(GUARD_PX), 0.0, 1.0).astype(np.float32)


# =========================================================================== #
#                                 三个通道
# =========================================================================== #
def _ch_global_shape(detrended: np.ndarray, mask: np.ndarray,
                     weight: np.ndarray, sigma_local: np.ndarray) -> np.ndarray:
    norm = _local_norm(detrended, mask, sigma_local) * weight
    out = ((norm + 1.0) * 127.5).astype(np.uint8)
    out[mask == 0] = 0
    return out


def _ch_defect(detrended: np.ndarray, mask: np.ndarray, weight: np.ndarray,
               sigma_local: np.ndarray) -> np.ndarray:
    """Top-Hat + LoG 多尺度融合的缺陷增强（用局部尺度归一化）"""
    work = detrended.astype(np.float32)
    fill_val = float(np.median(work[mask > 0])) if (mask > 0).any() else 0.0
    work_view = np.where(mask > 0, work, fill_val).astype(np.float32)

    white_max = np.zeros_like(work_view)
    black_max = np.zeros_like(work_view)
    for r in TH_RADII:
        kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(work_view, cv2.MORPH_OPEN, kr)
        closed = cv2.morphologyEx(work_view, cv2.MORPH_CLOSE, kr)
        np.maximum(white_max, work_view - opened, out=white_max)
        np.maximum(black_max, closed - work_view, out=black_max)
    th_signed = white_max - black_max

    log_best = np.zeros_like(work_view)
    log_best_abs = np.zeros_like(work_view)
    for s in LOG_SIGMAS:
        nlog = cv2.Laplacian(cv2.GaussianBlur(work_view, (0, 0), s),
                             cv2.CV_32F, ksize=3) * (s * s)
        a = np.abs(nlog)
        rep = a > log_best_abs
        log_best[rep] = nlog[rep]
        log_best_abs[rep] = a[rep]
    log_signed = -log_best

    # Top-Hat / LoG 的响应幅度本身随噪声水平放大，用同一局部尺度归一化才公平
    fused = ALPHA_TH * _local_norm(th_signed, mask, sigma_local) + \
            ALPHA_LOG * _local_norm(log_signed, mask, sigma_local)
    fused = np.clip(fused, -1.0, 1.0) * weight
    out = ((fused + 1.0) * 127.5).astype(np.uint8)
    out[mask == 0] = 128
    return out


def _ch_intensity(intensity8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩膜内做 CLAHE，掩膜外填均值后置 0，背景保持纯净"""
    if (mask > 0).any():
        fill = int(np.clip(np.mean(intensity8[mask > 0]), 0, 255))
        src = np.where(mask > 0, intensity8, fill).astype(np.uint8)
    else:
        src = intensity8
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                            tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    out = clahe.apply(src)
    out[mask == 0] = 0
    return out


# =========================================================================== #
#                                 公开 API
# =========================================================================== #
def synthesize(depth16: np.ndarray, intensity8: np.ndarray) -> np.ndarray:
    """
    3D 数据 -> 3 通道 BGR uint8 训练图。

    两侧噪点通过「局部噪声尺度归一化 + 噪声驱动的自适应去噪」就地压制，
    工件宽度保持不变（不做任何列裁剪）。
    """
    depth16 = _ensure_depth16(depth16)
    intensity8 = _ensure_intensity8(intensity8)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8,
                                (depth16.shape[1], depth16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    raw_mask, mask = build_valid_mask(depth16)
    weight = _guard_weight(mask)

    depth_f = depth16.astype(np.float32)
    depth_filled = np.where(raw_mask > 0, depth_f,
                            _mask_aware_gaussian(depth_f, raw_mask, FILL_SIGMA))
    residual = depth_filled - _mask_aware_gaussian(depth_filled, raw_mask,
                                                   SIGMA_BASELINE)
    residual = _remove_row_bias(residual, mask)

    # 1) 估噪声尺度 -> 2) 只在高噪声区去噪 -> 3) 用更新后的尺度做局部归一化
    residual = _denoise_by_noise(residual, mask, _local_scale(residual, mask))
    sigma_local = _local_scale(residual, mask)

    ch_r = _ch_global_shape(residual, mask, weight, sigma_local)
    ch_g = _ch_defect(residual, mask, weight, sigma_local)
    ch_b = _ch_intensity(intensity8, mask)
    return cv2.merge([ch_b, ch_g, ch_r])


# =========================================================================== #
#                            文件名解析 / 配对
# =========================================================================== #
HEIGHT_TOKEN = "height"
INTENSITY_TOKEN = "intensity"

DEPTH_EXTS = (".tiff", ".tif")
INTENSITY_EXTS = (".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg")

# 排除干扰文件（伪彩图也是 .png，容易被误当亮度图）
EXCLUDE_TOKENS = ("pseudocolor", "pseudo_color", "color", "merged")


def _split_role(path: str, token: str):
    """
    从文件名剥离角色标记，返回 (前缀 stem, 是否命中)。
        img-0_height.tiff -> ("img-0", True)
    按 token 最后一次出现的位置切分，避免前缀里含同名单词时切错。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    idx = stem.lower().rfind(token)
    if idx < 0:
        return stem, False
    prefix = stem[:idx].rstrip("_- .")
    if not prefix:
        return stem, False
    return prefix, True


def _is_excluded(path: str) -> bool:
    low = os.path.basename(path).lower()
    return any(tok in low for tok in EXCLUDE_TOKENS)


def _iter_files(folder: str):
    pattern = os.path.join(folder, "**", "*") if RECURSIVE \
        else os.path.join(folder, "*")
    for p in glob(pattern, recursive=RECURSIVE):
        if os.path.isfile(p):
            yield p


def auto_pair(folder: str):
    """按 <前缀>_height.tiff 与 <前缀>_intensity.png 在同一目录内配对"""
    depth_map, inten_map = {}, {}

    for p in _iter_files(folder):
        ext = os.path.splitext(p)[1].lower()
        low_name = os.path.basename(p).lower()
        d = os.path.dirname(os.path.abspath(p))

        if HEIGHT_TOKEN in low_name and ext in DEPTH_EXTS:
            prefix, ok = _split_role(p, HEIGHT_TOKEN)
            if ok:
                depth_map[(d, prefix)] = p
            continue

        if INTENSITY_TOKEN in low_name and ext in INTENSITY_EXTS \
                and not _is_excluded(p):
            prefix, ok = _split_role(p, INTENSITY_TOKEN)
            if ok:
                inten_map[(d, prefix)] = p

    pairs, missing = [], []
    for key in sorted(depth_map):
        if key in inten_map:
            pairs.append((depth_map[key], inten_map[key]))
        else:
            missing.append(depth_map[key])

    orphan_inten = [inten_map[k] for k in sorted(inten_map) if k not in depth_map]
    return pairs, missing, orphan_inten


def make_out_name(depth_path: str, in_root: str) -> str:
    """输出名带相对子目录前缀，避免不同子目录下同名文件互相覆盖"""
    prefix, _ = _split_role(depth_path, HEIGHT_TOKEN)
    rel_dir = os.path.relpath(os.path.dirname(os.path.abspath(depth_path)),
                              os.path.abspath(in_root))
    if rel_dir in (".", ""):
        return f"{prefix}_merged.png"
    tag = rel_dir.replace(os.sep, "_").replace("/", "_").replace("..", "up")
    return f"{tag}__{prefix}_merged.png"


def unique_path(out_dir: str, name: str) -> str:
    """同名时追加 _1 / _2 ...，绝不覆盖已有结果"""
    stem, ext = os.path.splitext(name)
    cand = os.path.join(out_dir, name)
    n = 1
    while os.path.exists(cand):
        cand = os.path.join(out_dir, f"{stem}_{n}{ext}")
        n += 1
    return cand


# =========================================================================== #
#                                  批量 IO
# =========================================================================== #
def process_pair(depth_path: str, inten_path: str, out_dir: str,
                 in_root: str) -> str:
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(inten_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(f"无法读取高度图: {depth_path}")
    if intensity8 is None:
        raise FileNotFoundError(f"无法读取亮度图: {inten_path}")
    if depth16.ndim == 3:                       # 少数 tiff 带多通道，取第一通道
        depth16 = depth16[:, :, 0]

    merged = synthesize(depth16, intensity8)

    out_path = unique_path(out_dir, make_out_name(depth_path, in_root))
    if not cv2.imwrite(out_path, merged):
        raise IOError(f"写出失败: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="批量 3D 高度图(_height) + 亮度图(_intensity) -> 3 通道训练图"
                    "（两侧噪点就地抑制，不裁剪工件）")
    p.add_argument("--input", "-i",
                   default=r"D:\E\github_zl\3D_image_synthesis_methods_master\imgs\111\0813CS\32-16-130",
                   help="输入文件夹（含 *_height.tiff 与 *_intensity.png）")
    p.add_argument("--output", "-o", default="./results-fast-hi-edgeclean",
                   help="输出文件夹")
    args = p.parse_args()

    in_root = args.input
    if not os.path.isdir(in_root):
        print(f"[!] 输入目录不存在: {in_root}")
        return

    os.makedirs(args.output, exist_ok=True)

    pairs, missing, orphan = auto_pair(in_root)
    if not pairs:
        print(f"[!] 未在 {in_root} 找到 *_height.tiff / *_intensity.png 配对")
        return

    print(f"[i] 配对成功 {len(pairs)} 组"
          f"（缺亮度图 {len(missing)}，缺高度图 {len(orphan)}）")
    for m in missing:
        print(f"    [skip] 缺 *_{INTENSITY_TOKEN}: {m}")
    for o in orphan:
        print(f"    [skip] 缺 *_{HEIGHT_TOKEN}: {o}")

    ok = 0
    for t, png in tqdm(pairs, desc="synthesize"):
        try:
            out = process_pair(t, png, args.output, in_root)
            tqdm.write(f"  {os.path.basename(t)} + {os.path.basename(png)}"
                       f" -> {os.path.basename(out)}")
            ok += 1
        except Exception as e:
            tqdm.write(f"[ERR] {t}: {e}")

    print(f"[OK] 完成 {ok}/{len(pairs)}，输出目录: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

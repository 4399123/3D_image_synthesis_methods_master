"""
批量版（height / intensity）+ 两侧边缘噪点抑制（不裁剪工件）—— 加速版。

与 synthesize-多图-西克-去噪版.py 的**数值结果完全一致（逐像素 bit-identical）**，
只是把重计算限制在必要区域内，不改动任何算法与参数。

加速依据（实测 5000x3200，单图 6.34s 的耗时分布）
--------------------------------------------------
    ch_defect(G)      4.693s   <- 74%，其中形态学 3.48s（r=21 单独 2.44s）
    denoise           0.298s
    local_scale x2    0.522s
    ch_global(R)      0.224s
    其余(mask/基线/行偏置/亮度) 约 0.6s

工件掩膜的包围盒只占全图 **31.3%**：ys 1842~5000, xs 996~2582。而掩膜之外的
输出本来就是常数（R=0、G=128、B=0），完全不必参与形态学/LoG/归一化运算。

因此本版做两件事，都不改变数值：
  1. **包围盒裁剪重计算**：形态学、LoG、局部归一化、自适应去噪只在
     `bbox + PAD` 子图上做，算完贴回。PAD = 4*max(TH_RADII) = 84 px，
     远大于形态学结构元支撑(3*21=63，已实测 bit-identical)与
     LoG 高斯支撑(~4*3=12)，保证掩膜内每个像素的邻域都完整，结果精确相等。
     子图边缘若与原图边界重合，边界处理方式也与原图一致。
  2. **去掉重复的全局统计**：_local_norm 每次调用都要 `x[mask>0]` 抽取 480 万
     元素再求 median/MAD，改为在子图上抽取（掩膜像素全在包围盒内，取值集合
     完全相同，中位数/MAD 逐位相等）。

为什么**不**裁剪这几步（会破坏一致性，故保持全图）：
  * _mask_aware_gaussian 在 sigma>=16 时走「下采样卷积」路径，
    resize 目标尺寸取决于全图 W,H；裁剪会改变采样网格 -> 结果不再逐位相同。
    故 FILL_SIGMA/SIGMA_BASELINE/SCALE_SIGMA 相关步骤全部保持全图。
  * CLAHE 的 tile 网格由图像尺寸决定，裁剪会改变分块 -> 亮度通道保持全图。
  这些步骤合计仅约 0.9s，不是瓶颈。

另外：残差在掩膜之外的取值对最终结果没有影响（_mask_aware_gaussian 会乘掩膜、
两个通道又都对 mask==0 强制赋常数），因此自适应去噪只在包围盒内做同样安全。

用法：
    python synthesize-fast-多图-西克-去噪版.py -i ./imgs -o ./results
    python synthesize-fast-多图-西克-去噪版.py --benchmark   # 与原版对比速度并校验一致性
"""

import os
import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================================== #
#                    固定参数（与原版完全一致，不对外开放）
# =========================================================================== #
SIGMA_BASELINE = 60.0     # 低频基线高斯 sigma
ERODE_PX = 8              # 有效区内缩像素
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

# 包围盒外扩像素：必须 >= 形态学结构元支撑，实测 3*21=63 即已逐位相等，取 84 更保险
PAD = 4 * max(TH_RADII)

# 批量并行：单图内部 OpenCV 已多线程，但线程扩展性有限；实测 20 核机器上
# 「10 进程 x 每进程 2 线程」比「串行 + 满线程」再快 3.65 倍。
WORKERS = min(10, max(1, (os.cpu_count() or 4) // 2))
THREADS_PER_WORKER = 2


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
    逐像素噪声尺度 sigma_local（保持全图计算）。

    这里两次 _mask_aware_gaussian 的 sigma 分别为 FINE_SIGMA=2（普通卷积）与
    SCALE_SIGMA=48（下采样路径，网格依赖全图尺寸），因此不做裁剪以保证逐位一致。
    """
    dev = np.abs(res - _mask_aware_gaussian(res, mask, FINE_SIGMA))
    scale = _mask_aware_gaussian(dev, mask, SCALE_SIGMA)
    return np.maximum(scale, 1e-6).astype(np.float32)


def _denoise_by_noise(res: np.ndarray, mask: np.ndarray,
                      sigma_local: np.ndarray, box=None) -> np.ndarray:
    """
    空间自适应去噪：只在噪声比值高的地方（两侧）平滑，中部保持原样。

    box 给定时只在子图上做中值/高斯并原位贴回：掩膜外的残差不影响任何输出
    （后续两个通道都对 mask==0 强制赋常数，_mask_aware_gaussian 也会乘掩膜），
    因此结果与全图计算在掩膜内逐位相同。
    quiet 分位数仍按掩膜内全部像素统计（取值集合不变）。
    """
    if NR_STRENGTH <= 0:
        return res

    v = sigma_local[mask > 0]
    quiet = max(float(np.percentile(v, 35)), 1e-6) if v.size else 1.0

    if box is None:
        sl = (slice(None), slice(None))
    else:
        y0, y1, x0, x1 = box
        sl = (slice(y0, y1), slice(x0, x1))

    sub = res[sl]
    ratio = sigma_local[sl] / quiet
    w = np.clip((ratio - RATIO_LO) / max(1e-6, RATIO_HI - RATIO_LO), 0.0, 1.0)
    w = np.clip(w * NR_STRENGTH, 0.0, 1.0).astype(np.float32)

    smooth = cv2.medianBlur(np.ascontiguousarray(sub, dtype=np.float32),
                            int(MED_KSIZE) | 1)
    if BLUR_SIGMA > 0:
        smooth = cv2.GaussianBlur(smooth, (0, 0), BLUR_SIGMA)

    out = res.copy()
    out[sl] = ((1.0 - w) * sub + w * smooth).astype(np.float32)
    return out


def _mask_stats(x: np.ndarray, mask: np.ndarray):
    """掩膜内的 median 与 MAD（供局部归一化用）"""
    v = x[mask > 0]
    if v.size == 0:
        return None, None
    med = float(np.median(v))
    gmad = float(np.median(np.abs(v - med)))
    if gmad < 1e-3:
        gmad = max(1e-3, 0.05 * float(np.std(v)))
    return med, gmad


def _local_norm_sub(x: np.ndarray, mask: np.ndarray,
                    sigma_local: np.ndarray) -> np.ndarray:
    """
    局部归一化到 [-1,1]：x / (K_LOCAL * max(sigma_local, floor))。

    传入的都是**子图**。掩膜像素全部落在包围盒内，所以 median/MAD 的取值集合
    与全图统计完全相同，结果逐位相等。
    """
    med, gmad = _mask_stats(x, mask)
    if med is None:
        return np.zeros_like(x)
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
    """返回 (raw_mask, mask)；ERODE_PX=8，工件宽度与原版一致"""
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


def _mask_box(mask: np.ndarray):
    """
    掩膜包围盒外扩 PAD，返回 (y0, y1, x0, x1)。

    用列/行投影求边界，比 np.where 拿全部坐标更省内存。
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(0, int(rows[0]) - PAD)
    y1 = min(mask.shape[0], int(rows[-1]) + 1 + PAD)
    x0 = max(0, int(cols[0]) - PAD)
    x1 = min(mask.shape[1], int(cols[-1]) + 1 + PAD)
    return y0, y1, x0, x1


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
                     weight: np.ndarray, sigma_local: np.ndarray,
                     box) -> np.ndarray:
    """R 通道：只在包围盒内归一化，盒外掩膜必为 0 -> 输出恒为 0"""
    y0, y1, x0, x1 = box
    sl = (slice(y0, y1), slice(x0, x1))
    m_sub = mask[sl]

    norm = _local_norm_sub(detrended[sl], m_sub, sigma_local[sl]) * weight[sl]
    sub = ((norm + 1.0) * 127.5).astype(np.uint8)
    sub[m_sub == 0] = 0

    out = np.zeros(mask.shape, dtype=np.uint8)      # 盒外即 mask==0 -> 0
    out[sl] = sub
    return out


def _ch_defect(detrended: np.ndarray, mask: np.ndarray, weight: np.ndarray,
               sigma_local: np.ndarray, box) -> np.ndarray:
    """
    G 通道：Top-Hat + LoG 多尺度融合（全部在包围盒子图上做）。

    盒外掩膜恒为 0，原版对 mask==0 直接赋 128，故盒外整片填 128 即可，
    与原版逐位相同。PAD=84 保证盒内每个掩膜像素的形态学/LoG 邻域完整。
    """
    y0, y1, x0, x1 = box
    sl = (slice(y0, y1), slice(x0, x1))
    m_sub = mask[sl]

    work = detrended[sl].astype(np.float32)
    fill_val = float(np.median(work[m_sub > 0])) if (m_sub > 0).any() else 0.0
    work_view = np.where(m_sub > 0, work, fill_val).astype(np.float32)
    work_view = np.ascontiguousarray(work_view)

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

    s_sub = sigma_local[sl]
    fused = ALPHA_TH * _local_norm_sub(th_signed, m_sub, s_sub) + \
            ALPHA_LOG * _local_norm_sub(log_signed, m_sub, s_sub)
    fused = np.clip(fused, -1.0, 1.0) * weight[sl]
    sub = ((fused + 1.0) * 127.5).astype(np.uint8)
    sub[m_sub == 0] = 128

    out = np.full(mask.shape, 128, dtype=np.uint8)  # 盒外即 mask==0 -> 128
    out[sl] = sub
    return out


def _ch_intensity(intensity8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    B 通道：掩膜内做 CLAHE，掩膜外置 0。

    CLAHE 的 tile 网格由图像尺寸决定，裁剪会改变分块结果，故保持全图（仅 0.05s）。
    """
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
    """3D 数据 -> 3 通道 BGR uint8 训练图（与原版逐像素一致）"""
    depth16 = _ensure_depth16(depth16)
    intensity8 = _ensure_intensity8(intensity8)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8,
                                (depth16.shape[1], depth16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    raw_mask, mask = build_valid_mask(depth16)
    box = _mask_box(mask)                      # 后续重活都限制在这个盒子里
    weight = _guard_weight(mask)

    depth_f = depth16.astype(np.float32)
    depth_filled = np.where(raw_mask > 0, depth_f,
                            _mask_aware_gaussian(depth_f, raw_mask, FILL_SIGMA))
    residual = depth_filled - _mask_aware_gaussian(depth_filled, raw_mask,
                                                   SIGMA_BASELINE)
    residual = _remove_row_bias(residual, mask)

    # 1) 估噪声尺度 -> 2) 只在高噪声区去噪 -> 3) 用更新后的尺度做局部归一化
    residual = _denoise_by_noise(residual, mask,
                                 _local_scale(residual, mask), box)
    sigma_local = _local_scale(residual, mask)

    ch_r = _ch_global_shape(residual, mask, weight, sigma_local, box)
    ch_g = _ch_defect(residual, mask, weight, sigma_local, box)
    ch_b = _ch_intensity(intensity8, mask)
    return cv2.merge([ch_b, ch_g, ch_r])


# =========================================================================== #
#                            文件名解析 / 配对
# =========================================================================== #
HEIGHT_TOKEN = "height"
INTENSITY_TOKEN = "intensity"

DEPTH_EXTS = (".tiff", ".tif")
INTENSITY_EXTS = (".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg")

EXCLUDE_TOKENS = ("pseudocolor", "pseudo_color", "color", "merged")


def _split_role(path: str, token: str):
    """从文件名剥离角色标记：img-0_height.tiff -> ("img-0", True)"""
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
def read_pair(depth_path: str, inten_path: str):
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(inten_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(f"无法读取高度图: {depth_path}")
    if intensity8 is None:
        raise FileNotFoundError(f"无法读取亮度图: {inten_path}")
    if depth16.ndim == 3:                       # 少数 tiff 带多通道，取第一通道
        depth16 = depth16[:, :, 0]
    return depth16, intensity8


def process_pair(depth_path: str, inten_path: str, out_dir: str,
                 in_root: str, out_name: str = None) -> str:
    depth16, intensity8 = read_pair(depth_path, inten_path)
    merged = synthesize(depth16, intensity8)

    if out_name is None:                       # 串行路径：即时定名
        out_name = unique_path(out_dir, make_out_name(depth_path, in_root))
    if not cv2.imwrite(out_name, merged):
        raise IOError(f"写出失败: {out_name}")
    return out_name


def _worker(task):
    """
    子进程入口：每进程限制 OpenCV 线程数，避免 N 进程 x 满线程互相抢核。

    输出路径在主进程预先分配好（unique_path 依赖文件系统状态，并行下会竞争）。
    """
    depth_path, inten_path, out_path = task
    cv2.setNumThreads(THREADS_PER_WORKER)
    try:
        return process_pair(depth_path, inten_path, None, None, out_path), None
    except Exception as e:                     # 单图失败不影响整批
        return depth_path, str(e)


# =========================================================================== #
#                        与原版对比：速度 + 一致性
# =========================================================================== #
BASE_SCRIPT = "synthesize-多图-西克-去噪版.py"


def _load_baseline(path: str):
    """动态加载原版脚本（文件名含中文/连字符，不能直接 import）"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("baseline_syn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def benchmark(in_root: str, limit: int = 0):
    """逐图跑原版与加速版，比较耗时并校验逐像素一致"""
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             BASE_SCRIPT)
    if not os.path.isfile(base_path):
        print(f"[!] 找不到原版脚本，无法对比: {base_path}")
        return
    base = _load_baseline(base_path)

    pairs, _, _ = auto_pair(in_root)
    if not pairs:
        print(f"[!] 未在 {in_root} 找到配对")
        return
    if limit > 0:
        pairs = pairs[:limit]

    t_old_all, t_new_all, same_all = 0.0, 0.0, True
    print(f"{'file':22s} {'原版(s)':>9s} {'加速版(s)':>10s} {'加速比':>7s}  一致")
    for dp, ip in pairs:
        depth16, intensity8 = read_pair(dp, ip)

        t = time.perf_counter()
        a = base.synthesize(depth16, intensity8)
        t_old = time.perf_counter() - t

        t = time.perf_counter()
        b = synthesize(depth16, intensity8)
        t_new = time.perf_counter() - t

        same = a.shape == b.shape and np.array_equal(a, b)
        same_all &= same
        t_old_all += t_old
        t_new_all += t_new
        print(f"{os.path.basename(dp):22s} {t_old:9.3f} {t_new:10.3f} "
              f"{t_old / max(t_new, 1e-9):6.2f}x  {'是' if same else '否!!'}")

    n = len(pairs)
    print("-" * 64)
    print(f"合计 {n} 图: 原版 {t_old_all:.2f}s ({t_old_all/n:.2f}s/图), "
          f"加速版 {t_new_all:.2f}s ({t_new_all/n:.2f}s/图)")
    print(f"平均加速比 {t_old_all / max(t_new_all, 1e-9):.2f}x，"
          f"逐像素完全一致: {'是' if same_all else '否'}")


def main():
    p = argparse.ArgumentParser(
        description="批量 3D 高度图(_height) + 亮度图(_intensity) -> 3 通道训练图"
                    "（加速版，结果与原版逐像素一致）")
    p.add_argument("--input", "-i",
                   default=r"D:\E\github_zl\3D_image_synthesis_methods_master\imgs\111\0813CS\32-16-130",
                   help="输入文件夹（含 *_height.tiff 与 *_intensity.png）")
    p.add_argument("--output", "-o", default="./results-fast-hi-edgeclean_fast",
                   help="输出文件夹")
    p.add_argument("--benchmark", action="store_true",
                   help="与原版对比速度并校验一致性（不写出结果）")
    p.add_argument("--bench-limit", type=int, default=0,
                   help="对比时只跑前 N 图，0=全部")
    p.add_argument("--jobs", "-j", type=int, default=0,
                   help=f"并行进程数，0=自动({WORKERS})，1=串行")
    args = p.parse_args()

    in_root = args.input
    if not os.path.isdir(in_root):
        print(f"[!] 输入目录不存在: {in_root}")
        return

    if args.benchmark:
        benchmark(in_root, args.bench_limit)
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

    # 输出路径在主进程串行分配，保证并行下不重名、不竞争
    tasks = []
    taken = set()
    for dp, ip in pairs:
        name = make_out_name(dp, in_root)
        op = unique_path(args.output, name)
        while op in taken:                     # 同批内再去重
            stem, ext = os.path.splitext(op)
            op = f"{stem}_x{ext}"
        taken.add(op)
        tasks.append((dp, ip, op))

    workers = 1 if args.jobs == 1 else (args.jobs or WORKERS)
    workers = max(1, min(workers, len(tasks)))

    ok, errs = 0, []
    if workers == 1:
        cv2.setNumThreads(0)                   # 串行时放开全部线程
        for dp, ip, op in tqdm(tasks, desc="synthesize"):
            try:
                process_pair(dp, ip, None, None, op)
                ok += 1
            except Exception as e:
                errs.append((dp, str(e)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res, err in tqdm(ex.map(_worker, tasks), total=len(tasks),
                                 desc=f"synthesize x{workers}"):
                if err is None:
                    ok += 1
                else:
                    errs.append((res, err))

    for dp, e in errs:
        print(f"[ERR] {dp}: {e}")
    print(f"[OK] 完成 {ok}/{len(pairs)}，输出目录: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

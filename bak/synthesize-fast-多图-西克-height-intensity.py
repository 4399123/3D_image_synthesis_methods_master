"""
批量版（height / intensity 命名规则）：

递归扫描输入目录，按 "同前缀 + 后缀" 规则配对：
    <stem>_height.tiff   (或 .tif)   -> 高度图 (uint16)
    <stem>_intensity.png (或 .tiff)  -> 亮度图 (uint8)
其余文件（如 <stem>_pseudocolor.png、.dat、.xml）一律忽略。

输出文件名带上相对子目录，避免不同子目录下的同名文件互相覆盖：
    imgs/31/img-0_height.tiff -> results/31__img-0_merged.png
    imgs/33/img-0_height.tiff -> results/33__img-0_merged.png
若仍然重名（极端情况），自动追加 _1 / _2 ... 后缀。

合成算法与 synthesize-fast-多图-西克.py 完全一致，未做改动。

用法：
    python synthesize-fast-多图-西克-height-intensity.py -i ./imgs -o ./results
    python ... -i ./imgs -o ./results --no-recursive     # 只扫描顶层目录
    python ... -i ./imgs -o ./results --flat             # 输出不带子目录前缀
"""

import os
import argparse
import warnings
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================================== #
#                            合成核心 (保持不变)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 输入清洗
# --------------------------------------------------------------------------- #
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
        return (i >> 8).astype(np.uint8)              # 缩放而非截断低位
    return np.clip(i, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _mask_aware_gaussian(img: np.ndarray, mask: np.ndarray,
                         sigma: float) -> np.ndarray:
    """掩膜感知高斯：blur(img*mask)/blur(mask)
    sigma>=16 时下采样卷积加速；sigma<16 直接卷积。
    """
    img_f = img.astype(np.float32) * mask.astype(np.float32)
    m_f = mask.astype(np.float32)

    if sigma >= 16.0:
        scale = max(1, int(round(sigma / 8.0)))
        H, W = img.shape
        ds = (W // scale, H // scale)
        img_s = cv2.resize(img_f, ds, interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(m_f, ds, interpolation=cv2.INTER_AREA)
        n_s = cv2.GaussianBlur(img_s, (0, 0), sigma / scale)
        d_s = cv2.GaussianBlur(m_s, (0, 0), sigma / scale) + 1e-6
        out_s = n_s / d_s
        return cv2.resize(out_s, (W, H), interpolation=cv2.INTER_LINEAR)

    n = cv2.GaussianBlur(img_f, (0, 0), sigma)
    d = cv2.GaussianBlur(m_f, (0, 0), sigma) + 1e-6
    return n / d


def _mad_norm(x: np.ndarray, mask: np.ndarray, k_mad: float) -> np.ndarray:
    """MAD 归一化到 [-1,1]；mad 近似为 0 时退化到 std 防止噪声被放大"""
    v = x[mask > 0]
    if v.size == 0:
        return np.zeros_like(x)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad < 1e-3:
        mad = max(1e-3, 0.05 * float(np.std(v)))
    return np.clip((x - med) / (k_mad * mad), -1.0, 1.0)


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


# --------------------------------------------------------------------------- #
# 三个通道
# --------------------------------------------------------------------------- #
def _ch_global_shape(detrended: np.ndarray, mask: np.ndarray,
                     k_mad: float = 6.0) -> np.ndarray:
    out = ((_mad_norm(detrended, mask, k_mad) + 1.0) * 127.5).astype(np.uint8)
    out[mask == 0] = 0
    return out


def _ch_defect(detrended: np.ndarray, mask: np.ndarray,
               th_radii=(5, 11, 21),
               log_sigmas=(1.2, 2.0, 3.0),
               k_mad: float = 4.0,
               alpha_th: float = 0.6,
               alpha_log: float = 0.4) -> np.ndarray:
    """Top-Hat + LoG 多尺度融合的缺陷增强"""
    work = detrended.astype(np.float32)
    fill_val = float(np.median(work[mask > 0])) if (mask > 0).any() else 0.0
    work_view = np.where(mask > 0, work, fill_val).astype(np.float32)

    white_max = np.zeros_like(work_view)
    black_max = np.zeros_like(work_view)
    for r in th_radii:
        kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(work_view, cv2.MORPH_OPEN, kr)
        closed = cv2.morphologyEx(work_view, cv2.MORPH_CLOSE, kr)
        np.maximum(white_max, work_view - opened, out=white_max)
        np.maximum(black_max, closed - work_view, out=black_max)
    th_signed = white_max - black_max

    log_best = np.zeros_like(work_view)
    log_best_abs = np.zeros_like(work_view)
    for s in log_sigmas:
        nlog = cv2.Laplacian(cv2.GaussianBlur(work_view, (0, 0), s),
                             cv2.CV_32F, ksize=3) * (s * s)
        a = np.abs(nlog)
        rep = a > log_best_abs
        log_best[rep] = nlog[rep]
        log_best_abs[rep] = a[rep]
    log_signed = -log_best

    fused = alpha_th * _mad_norm(th_signed, mask, k_mad) + \
            alpha_log * _mad_norm(log_signed, mask, k_mad)
    fused = np.clip(fused, -1.0, 1.0)
    out = ((fused + 1.0) * 127.5).astype(np.uint8)
    out[mask == 0] = 128
    return out


def _ch_intensity(intensity8: np.ndarray,
                  clip: float = 3.0, tile: int = 16) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(intensity8)


# --------------------------------------------------------------------------- #
# 公开 API
# --------------------------------------------------------------------------- #
def synthesize(depth16: np.ndarray, intensity8: np.ndarray,
               sigma_baseline: float = 60.0,
               erode_px: int = 8) -> np.ndarray:
    """3D 数据 -> 3 通道 BGR uint8 训练图"""
    depth16 = _ensure_depth16(depth16)
    intensity8 = _ensure_intensity8(intensity8)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8,
                                (depth16.shape[1], depth16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    raw_mask = (depth16 > 0).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode_px + 1,) * 2)
    mask = cv2.erode(raw_mask, se, iterations=1)

    depth_f = depth16.astype(np.float32)
    depth_filled = np.where(raw_mask > 0, depth_f,
                            _mask_aware_gaussian(depth_f, raw_mask, 24.0))
    residual = depth_filled - _mask_aware_gaussian(depth_filled, raw_mask,
                                                  sigma_baseline)
    residual = _remove_row_bias(residual, mask)

    ch_r = _ch_global_shape(residual, mask)
    ch_g = _ch_defect(residual, mask)
    ch_b = _ch_intensity(intensity8)
    return cv2.merge([ch_b, ch_g, ch_r])


# =========================================================================== #
#                     文件名解析 / 配对（本脚本的改动点）
# =========================================================================== #
HEIGHT_TOKEN = "height"
INTENSITY_TOKEN = "intensity"

DEPTH_EXTS = (".tiff", ".tif")
INTENSITY_EXTS = (".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg")

# 需要显式排除的干扰文件（如伪彩图，它也是 .png，很容易被误当亮度图）
EXCLUDE_TOKENS = ("pseudocolor", "pseudo_color", "color", "merged")


def _split_role(path: str, token: str):
    """
    从文件名中剥离角色标记，返回 (前缀 stem, 是否命中)。

    形如 <stem>_height / <stem>-height / <stem>height 都能识别，
    同时按 token 出现的最后一次位置切分，避免前缀里恰好含有同名单词时切错。
        img-0_height.tiff        -> ("img-0", True)
        height_scan_1_height.tif -> ("height_scan_1", True)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    low = stem.lower()
    idx = low.rfind(token)
    if idx < 0:
        return stem, False
    prefix = stem[:idx].rstrip("_- .")
    if not prefix:                      # 文件名就叫 height.tiff，没有可配对的前缀
        return stem, False
    return prefix, True


def _is_excluded(path: str) -> bool:
    low = os.path.basename(path).lower()
    return any(tok in low for tok in EXCLUDE_TOKENS)


def _iter_files(folder: str, recursive: bool):
    pattern = os.path.join(folder, "**", "*") if recursive \
        else os.path.join(folder, "*")
    for p in glob(pattern, recursive=recursive):
        if os.path.isfile(p):
            yield p


def auto_pair(folder: str, recursive: bool = True):
    """
    按 <前缀>_height.<tiff|tif> 与 <前缀>_intensity.<png|...> 配对。

    只在同一目录内配对（不同子目录的同名前缀互不干扰）。
    返回 [(depth_path, intensity_path), ...]，按路径排序。
    """
    depth_map = {}       # (dir, prefix) -> depth path
    inten_map = {}       # (dir, prefix) -> intensity path

    for p in _iter_files(folder, recursive):
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

    orphan_inten = [inten_map[k] for k in sorted(inten_map)
                    if k not in depth_map]
    return pairs, missing, orphan_inten


# --------------------------------------------------------------------------- #
# 输出命名：把相对子目录写进文件名，防止跨目录同名覆盖
# --------------------------------------------------------------------------- #
def make_out_name(depth_path: str, in_root: str, flat: bool = False) -> str:
    """
    imgs/31/img-0_height.tiff, in_root=imgs -> "31__img-0_merged.png"
    flat=True 时 -> "img-0_merged.png"
    """
    prefix, _ = _split_role(depth_path, HEIGHT_TOKEN)

    if flat:
        return f"{prefix}_merged.png"

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
#                                批量 IO
# =========================================================================== #
def process_pair(depth_path: str, inten_path: str, out_dir: str,
                 in_root: str, flat: bool = False,
                 overwrite: bool = False) -> str:
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(inten_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(f"无法读取高度图: {depth_path}")
    if intensity8 is None:
        raise FileNotFoundError(f"无法读取亮度图: {inten_path}")
    if depth16.ndim == 3:                       # 少数 tiff 带多通道，取第一通道
        depth16 = depth16[:, :, 0]

    merged = synthesize(depth16, intensity8)

    name = make_out_name(depth_path, in_root, flat)
    out_path = os.path.join(out_dir, name) if overwrite \
        else unique_path(out_dir, name)
    if not cv2.imwrite(out_path, merged):
        raise IOError(f"写出失败: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="批量 3D 高度图(_height) + 亮度图(_intensity) -> 3 通道训练图")
    p.add_argument("--input", "-i", default=r"D:\E\github_zl\3D_image_synthesis_methods_master\imgs\814-33",
                   help="输入文件夹（含 *_height.tiff 与 *_intensity.png）")
    p.add_argument("--output", "-o", default="./results-fast-hi",
                   help="输出文件夹")
    p.add_argument("--no-recursive", action="store_true",
                   help="只扫描顶层目录，不递归子目录")
    p.add_argument("--flat", action="store_true",
                   help="输出文件名不带子目录前缀（重名时靠 _1/_2 区分）")
    p.add_argument("--overwrite", action="store_true",
                   help="允许覆盖同名输出（默认不覆盖，自动改名）")
    args = p.parse_args()

    in_root = args.input
    if not os.path.isdir(in_root):
        print(f"[!] 输入目录不存在: {in_root}")
        return

    os.makedirs(args.output, exist_ok=True)

    pairs, missing, orphan = auto_pair(in_root, recursive=not args.no_recursive)
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
            out = process_pair(t, png, args.output, in_root,
                               flat=args.flat, overwrite=args.overwrite)
            tqdm.write(f"  {os.path.basename(t)} + {os.path.basename(png)}"
                       f" -> {os.path.basename(out)}")
            ok += 1
        except Exception as e:
            tqdm.write(f"[ERR] {t}: {e}")

    print(f"[✓] 完成 {ok}/{len(pairs)}，输出目录: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

"""
批量版：扫描输入目录里的同名 .tiff/.tif + .png 配对，
逐对合成 3 通道训练图，写到输出目录。

(合成实现内联自 synthesize_pack_fast.py，无外部依赖)

用法：
    python synthesize_batch.py -i ./imgs -o ./results
"""

import os
import argparse
import warnings
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================================== #
#                            合成核心 (来自 synthesize_pack_fast.py)
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
#                            批量 IO
# =========================================================================== #
def auto_pair(folder: str):
    """同目录下同名 .tiff/.tif + .png 的配对"""
    tiffs = sorted(glob(os.path.join(folder, "*.tiff")) +
                   glob(os.path.join(folder, "*.tif")))
    pairs = []
    for t in tiffs:
        stem = os.path.splitext(os.path.basename(t))[0]
        png = os.path.join(folder, stem + ".png")
        if os.path.exists(png):
            pairs.append((t, png))
    return pairs


def process_pair(depth_path: str, png_path: str, out_dir: str) -> str:
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(depth_path)
    if intensity8 is None:
        raise FileNotFoundError(png_path)

    merged = synthesize(depth16, intensity8)

    name = os.path.splitext(os.path.basename(depth_path))[0]
    out_path = os.path.join(out_dir, f"{name}_merged.png")
    cv2.imwrite(out_path, merged)
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="批量 3D 高度图 + 亮度图 -> 3 通道训练图")
    p.add_argument("--input", "-i", default=r"./imgs1",
                   help="输入文件夹 (含同名 .tiff/.tif + .png)")
    p.add_argument("--output", "-o", default="./results-fast",
                   help="输出文件夹")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pairs = auto_pair(args.input)
    if not pairs:
        print(f"[!] 未在 {args.input} 找到成对的 .tiff/.tif + .png")
        return

    for t, png in tqdm(pairs, desc="synthesize"):
        try:
            out = process_pair(t, png, args.output)
            tqdm.write(f"  -> {out}")
        except Exception as e:
            tqdm.write(f"[ERR] {t}: {e}")


if __name__ == "__main__":
    main()

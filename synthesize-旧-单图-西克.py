"""
3D 高度图 + 亮度图 -> 3 通道训练图 (封装版)

输出 (BGR)：
    B = 强度 CLAHE        表面纹理
    G = 缺陷增强 (带符号)  凸>128, 凹<128, Top-Hat + LoG 多算子融合
    R = 去趋势后的形貌    消除工件本体曲率
"""

import os
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _mask_aware_gaussian(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """掩膜感知高斯：blur(img*mask)/blur(mask)"""
    img_f = img.astype(np.float32) * mask
    m_f = mask.astype(np.float32)
    num = cv2.GaussianBlur(img_f, (0, 0), sigma)
    den = cv2.GaussianBlur(m_f, (0, 0), sigma) + 1e-6
    return num / den


def _fill_invalid(depth_f: np.ndarray, mask: np.ndarray, sigma: float = 24.0) -> np.ndarray:
    filled = _mask_aware_gaussian(depth_f, mask, sigma)
    return np.where(mask > 0, depth_f, filled)


def _erode_mask(mask: np.ndarray, px: int = 8) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k, iterations=1)


def _remove_row_bias(residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """每行减去其内部中位数，干掉线扫水平条带噪声"""
    out = residual.copy()
    for r in range(out.shape[0]):
        m = mask[r] > 0
        if m.sum() > 32:
            out[r] -= np.median(residual[r, m])
    return out


def _mad_norm(x: np.ndarray, mask: np.ndarray, k_mad: float) -> np.ndarray:
    """MAD 归一化到 [-1, 1]"""
    v = x[mask > 0]
    if v.size == 0:
        return np.zeros_like(x)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    return np.clip((x - med) / (k_mad * mad), -1.0, 1.0)


# --------------------------------------------------------------------------- #
# 三个通道
# --------------------------------------------------------------------------- #
def _ch_global_shape(detrended: np.ndarray, mask: np.ndarray,
                     k_mad: float = 6.0) -> np.ndarray:
    norm = _mad_norm(detrended, mask, k_mad)
    out = ((norm + 1.0) * 127.5).astype(np.uint8)
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

    # 多尺度 Top-Hat
    white_max = np.zeros_like(work_view)
    black_max = np.zeros_like(work_view)
    for r in th_radii:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(work_view, cv2.MORPH_OPEN, k)
        closed = cv2.morphologyEx(work_view, cv2.MORPH_CLOSE, k)
        white_max = np.maximum(white_max, work_view - opened)
        black_max = np.maximum(black_max, closed - work_view)
    th_signed = white_max - black_max  # 凸 +, 凹 -

    # 多尺度 LoG (scale-normalized)
    log_best = np.zeros_like(work_view)
    log_best_abs = np.zeros_like(work_view)
    for s in log_sigmas:
        smoothed = cv2.GaussianBlur(work_view, (0, 0), s)
        nlog = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3) * (s ** 2)
        a = np.abs(nlog)
        replace = a > log_best_abs
        log_best[replace] = nlog[replace]
        log_best_abs[replace] = a[replace]
    log_signed = -log_best  # 凸缺陷 LoG 中心为负 -> 取负让凸为正

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
# 封装入口
# --------------------------------------------------------------------------- #
def synthesize(depth16: np.ndarray, intensity8: np.ndarray,
               sigma_baseline: float = 60.0,
               erode_px: int = 8) -> np.ndarray:
    """3D 数据 -> 3 通道 BGR uint8 训练图

    Args:
        depth16   : (H, W) uint16, 0 表示无效像素
        intensity8: (H, W) uint8
        sigma_baseline: 工件本体估计尺度（应大于最大缺陷半径）
        erode_px: mask 收缩像素数，避开边界跳变
    Returns:
        (H, W, 3) uint8, BGR 排列
    """
    if depth16.dtype != np.uint16:
        depth16 = depth16.astype(np.uint16)
    if intensity8.dtype != np.uint8:
        intensity8 = intensity8.astype(np.uint8)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8,
                                (depth16.shape[1], depth16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    raw_mask = (depth16 > 0).astype(np.uint8)
    mask = _erode_mask(raw_mask, erode_px)

    depth_f = depth16.astype(np.float32)
    depth_filled = _fill_invalid(depth_f, raw_mask, sigma=24.0)

    baseline = _mask_aware_gaussian(depth_filled, raw_mask, sigma_baseline)
    residual = depth_filled - baseline
    residual = _remove_row_bias(residual, mask)

    ch_r = _ch_global_shape(residual, mask)
    ch_g = _ch_defect(residual, mask)
    ch_b = _ch_intensity(intensity8)

    return cv2.merge([ch_b, ch_g, ch_r])


def synthesize_from_path(depth_path: str, png_path: str) -> np.ndarray:
    """从文件路径直接合成"""
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(depth_path)
    if intensity8 is None:
        raise FileNotFoundError(png_path)
    return synthesize(depth16, intensity8)


# --------------------------------------------------------------------------- #
# 示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    depth_path = r'./imgs/5.tiff'
    png_path = r'./imgs/5.png'
    output_dir = r'./results/'

    os.makedirs(output_dir, exist_ok=True)
    merged = synthesize_from_path(depth_path, png_path)

    name = os.path.splitext(os.path.basename(depth_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{name}_merged.png"), merged)

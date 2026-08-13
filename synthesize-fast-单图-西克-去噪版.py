"""
单图版：3D 高度图 + 亮度图 -> 3 通道 BGR 训练图（两侧边缘噪点抑制，不裁剪工件）

算法与 synthesize-fast-多图-西克-去噪版.py 完全一致（逐像素相同），
只是把批量扫描/配对/并行改成「指定单张输入」的用法，与
bak/synthesize-fast-单图-西克.py 的调用方式保持一致：
在文件末尾直接写死 depth_path / png_path / output_dir 即可运行。

核心思路
--------
两侧麻点带的根因不是掉点（掉点率 0.0%），而是掠射角处高度值抖动大
（局部粗糙度两侧约 165、中部约 70）。全局 MAD 归一化的尺度由面积占优的安静
中部决定，两侧较大的噪声被除以偏小的尺度后超出 ±1 而饱和，形成麻点。

因此用**局部噪声尺度**替代全局 MAD：
  1. _local_scale     逐像素噪声水平 sigma_local
  2. _denoise_by_noise 按噪声比值做空间自适应去噪（只动两侧，中部不变）
  3. _local_norm_sub  用 sigma_local 归一化，边缘噪声就地压回 ±1 内
工件宽度保持不变（不做任何列裁剪）。

性能：重计算（形态学 / LoG / 归一化）只在掩膜包围盒 + PAD 内进行，
      与全图计算逐位相同，单图约 1.4 倍加速。
"""

import os
import warnings

import cv2
import numpy as np


# =========================================================================== #
#                        固定参数（与多图版完全一致）
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

TH_RADII = (5, 11, 21)          # 缺陷通道 Top-Hat 多尺度半径
LOG_SIGMAS = (1.2, 2.0, 3.0)    # 缺陷通道 LoG 多尺度 sigma
ALPHA_TH = 0.6                  # Top-Hat 融合权重
ALPHA_LOG = 0.4                 # LoG 融合权重
CLAHE_CLIP = 3.0                # 亮度通道 CLAHE
CLAHE_TILE = 16

# 包围盒外扩像素：需 >= 形态学结构元支撑（3*21=63），取 84 更保险
PAD = 4 * max(TH_RADII)


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
        return (i >> 8).astype(np.uint8)              # 缩放而非截断低位
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
        warnings.simplefilter("ignore", RuntimeWarning)   # all-nan 行
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

    |res - blur(res, FINE_SIGMA)| 取高频幅度，再用大尺度平滑 (SCALE_SIGMA)
    得到平缓的噪声水平场：FINE_SIGMA 小 -> 不把缺陷的低频形状算成噪声；
    SCALE_SIGMA 大 -> 尺度场不被单个缺陷局部抬高（否则缺陷会自己把自己压掉）。
    """
    dev = np.abs(res - _mask_aware_gaussian(res, mask, FINE_SIGMA))
    scale = _mask_aware_gaussian(dev, mask, SCALE_SIGMA)
    return np.maximum(scale, 1e-6).astype(np.float32)


def _denoise_by_noise(res: np.ndarray, mask: np.ndarray,
                      sigma_local: np.ndarray, box) -> np.ndarray:
    """
    空间自适应去噪：只在噪声比值高的地方（两侧）平滑，中部保持原样。

    w = clip((ratio - LO) / (HI - LO), 0, 1) * NR_STRENGTH
    out = (1 - w) * res + w * smooth
    w 由噪声水平驱动而非位置驱动，所以不会误伤中部的真实缺陷。
    只在包围盒内计算：掩膜外的残差不影响任何输出（两个通道对 mask==0 都赋常数）。
    """
    if NR_STRENGTH <= 0:
        return res

    v = sigma_local[mask > 0]
    quiet = max(float(np.percentile(v, 35)), 1e-6) if v.size else 1.0

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


def _local_norm_sub(x: np.ndarray, mask: np.ndarray,
                    sigma_local: np.ndarray) -> np.ndarray:
    """
    局部归一化到 [-1,1]：x / (K_LOCAL * max(sigma_local, floor))。

    floor = FLOOR_FRAC * 全局 MAD，双向保护：
      * 防止过于安静的区域把微小起伏放大成假缺陷；
      * 保证同尺寸真实缺陷在中部依然有足够对比度。
    传入的是子图；掩膜像素全在包围盒内，故 median/MAD 与全图统计逐位相同。
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
    """返回 (raw_mask, mask)；ERODE_PX=8，不改变工件宽度"""
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
    """掩膜包围盒外扩 PAD，返回 (y0, y1, x0, x1)"""
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
    """R 通道：整体形状；盒外掩膜必为 0 -> 输出恒为 0"""
    y0, y1, x0, x1 = box
    sl = (slice(y0, y1), slice(x0, x1))
    m_sub = mask[sl]

    norm = _local_norm_sub(detrended[sl], m_sub, sigma_local[sl]) * weight[sl]
    sub = ((norm + 1.0) * 127.5).astype(np.uint8)
    sub[m_sub == 0] = 0

    out = np.zeros(mask.shape, dtype=np.uint8)
    out[sl] = sub
    return out


def _ch_defect(detrended: np.ndarray, mask: np.ndarray, weight: np.ndarray,
               sigma_local: np.ndarray, box) -> np.ndarray:
    """
    G 通道：Top-Hat + LoG 多尺度融合的缺陷增强（全部在包围盒子图上做）。

    盒外掩膜恒为 0，对应输出恒为 128，故整片预填 128 即可。
    PAD=84 保证盒内每个掩膜像素的形态学/LoG 邻域完整。
    """
    y0, y1, x0, x1 = box
    sl = (slice(y0, y1), slice(x0, x1))
    m_sub = mask[sl]

    work = detrended[sl].astype(np.float32)
    fill_val = float(np.median(work[m_sub > 0])) if (m_sub > 0).any() else 0.0
    work_view = np.ascontiguousarray(
        np.where(m_sub > 0, work, fill_val).astype(np.float32))

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
    s_sub = sigma_local[sl]
    fused = ALPHA_TH * _local_norm_sub(th_signed, m_sub, s_sub) + \
            ALPHA_LOG * _local_norm_sub(log_signed, m_sub, s_sub)
    fused = np.clip(fused, -1.0, 1.0) * weight[sl]
    sub = ((fused + 1.0) * 127.5).astype(np.uint8)
    sub[m_sub == 0] = 128

    out = np.full(mask.shape, 128, dtype=np.uint8)
    out[sl] = sub
    return out


def _ch_intensity(intensity8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    B 通道：掩膜内做 CLAHE，掩膜外置 0（背景保持纯净）。

    掩膜外先填成掩膜内均值，避免背景 0 挤压直方图。
    CLAHE 的 tile 网格由图像尺寸决定，故保持全图计算。
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
    """
    3D 数据 -> 3 通道 BGR uint8 训练图。

    Args:
        depth16   : (H, W) 16bit 高度图，0 表示无效像素
        intensity8: (H, W) 8bit 亮度图
    Returns:
        (H, W, 3) uint8, BGR
    """
    depth16 = _ensure_depth16(depth16)
    intensity8 = _ensure_intensity8(intensity8)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8,
                                (depth16.shape[1], depth16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    raw_mask, mask = build_valid_mask(depth16)
    box = _mask_box(mask)                      # 重活都限制在这个盒子里
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


def synthesize_from_path(depth_path: str, png_path: str) -> np.ndarray:
    """读取一对 高度图/亮度图 并合成"""
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(depth_path)
    if intensity8 is None:
        raise FileNotFoundError(png_path)
    if depth16.ndim == 3:                       # 少数 tiff 带多通道，取第一通道
        depth16 = depth16[:, :, 0]
    return synthesize(depth16, intensity8)


# =========================================================================== #
# 示例：直接修改下面三行为自己的路径
# =========================================================================== #
if __name__ == "__main__":
    depth_path = r'./imgs/111/0813CS/32-16-130/img-0_height.tiff'
    png_path = r'./imgs/111/0813CS/32-16-130/img-0_intensity.png'
    output_dir = r'./results/'

    os.makedirs(output_dir, exist_ok=True)
    merged = synthesize_from_path(depth_path, png_path)

    #保存结果图
    name = os.path.splitext(os.path.basename(depth_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{name}_merged.png"), merged)

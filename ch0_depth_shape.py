"""
单独提取 synthesize.py 里的 ch0 (depth_shape / 全局形貌) 通道

流程：
    1. 读 16bit 深度图 + 8bit 亮度图（亮度仅用于尺寸校验，不参与 ch0 计算）
    2. mask-aware 大尺度高斯估计工件本体并减去 (detrending)
    3. 行偏置消除：去掉 3D 线扫相机的水平条纹噪声
    4. MAD 归一化到 0~255
"""

import os
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
depth_path = r'./imgs/1.tiff'
png_path = r'./imgs/1.png'
output_dir = r'./results/'
sigma_fill = 24.0       # 无效区填充用的 mask-aware 高斯尺度
sigma_baseline = 60.0   # 工件本体估计尺度（应大于最大缺陷半径）
erode_px = 8            # mask 收缩像素，避开边界跳变
k_mad = 6.0             # 显示动态范围 = ±k_mad * MAD


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def mask_aware_gaussian(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """掩膜感知高斯：blur(img*mask) / blur(mask)，无效像素不污染滤波"""
    img_f = img.astype(np.float32) * mask
    m_f = mask.astype(np.float32)
    num = cv2.GaussianBlur(img_f, (0, 0), sigma)
    den = cv2.GaussianBlur(m_f, (0, 0), sigma) + 1e-6
    return num / den


def fill_invalid(depth_f: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """无效像素用邻域有效值的高斯加权填充"""
    filled = mask_aware_gaussian(depth_f, mask, sigma)
    return np.where(mask > 0, depth_f, filled)


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k, iterations=1)


def remove_row_bias(residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """每行减去其内部中位数，干掉 3D 线扫相机的水平条带噪声"""
    out = residual.copy()
    for r in range(out.shape[0]):
        m = mask[r] > 0
        if m.sum() > 32:
            out[r] -= np.median(residual[r, m])
    return out


def depth_shape_channel(detrended: np.ndarray, mask: np.ndarray,
                        k_mad: float = 6.0) -> np.ndarray:
    """ch0：去趋势后的高度，MAD 归一化到 0~255"""
    valid = detrended[mask > 0]
    if valid.size == 0:
        return np.zeros_like(detrended, dtype=np.uint8)
    med = np.median(valid)
    mad = np.median(np.abs(valid - med)) + 1e-6
    span = k_mad * mad
    norm = np.clip((detrended - med) / span, -1.0, 1.0)
    out = ((norm + 1.0) * 127.5).astype(np.uint8)
    out[mask == 0] = 0
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(output_dir, exist_ok=True)

    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(depth_path)
    if intensity8 is None:
        raise FileNotFoundError(png_path)
    if depth16.dtype != np.uint16:
        depth16 = depth16.astype(np.uint16)
    if depth16.shape != intensity8.shape:
        # 仅做尺寸一致性提示；ch0 本身不用 intensity
        print(f"[warn] depth {depth16.shape} vs intensity {intensity8.shape}")

    raw_mask = (depth16 > 0).astype(np.uint8)
    mask = erode_mask(raw_mask, erode_px)

    depth_f = depth16.astype(np.float32)
    depth_filled = fill_invalid(depth_f, raw_mask, sigma_fill)

    # 1) detrending
    baseline = mask_aware_gaussian(depth_filled, raw_mask, sigma_baseline)
    residual = depth_filled - baseline

    # 2) 行偏置消除
    residual = remove_row_bias(residual, mask)

    # 3) ch0 输出
    ch0 = depth_shape_channel(residual, mask, k_mad=k_mad)

    name = os.path.splitext(os.path.basename(depth_path))[0]
    out_path = os.path.join(output_dir, f"{name}_ch0_depth_shape.png")
    cv2.imwrite(out_path, ch0)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

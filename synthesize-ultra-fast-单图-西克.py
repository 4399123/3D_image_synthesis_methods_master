import os
import cv2
import numpy as np
from time import time

cv2.setNumThreads(4)

# 算法超参数
SIGMA_BASELINE = 60.0
FILL_SIGMA = 24.0
GUARD_PX = 6
K_LOCAL = 6.0
PAD = 84

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
_K7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
_K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K_ERODE = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))


def _ensure_depth16(d: np.ndarray) -> np.ndarray:
    if d.ndim != 2:
        d = d[:, :, 0] if d.ndim == 3 else d
    if d.dtype == np.uint16:
        return d
    if d.dtype == np.uint8:
        return d.astype(np.uint16)
    return np.clip(d, 0, 65535).astype(np.uint16)


def _ensure_intensity8(i: np.ndarray) -> np.ndarray:
    if i.ndim != 2:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i
    if i.dtype == np.uint8:
        return i
    if i.dtype == np.uint16:
        return (i >> 8).astype(np.uint8)
    return np.clip(i, 0, 255).astype(np.uint8)


def _fast_mag_sub(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    scale = max(1, int(round(sigma / 8.0)))
    H_s, W_s = img.shape
    ds = (max(1, W_s // scale), max(1, H_s // scale))
    img_s = cv2.resize(img * mask, ds, interpolation=cv2.INTER_AREA)
    m_s = cv2.resize(mask, ds, interpolation=cv2.INTER_AREA)
    sig_s = sigma / scale
    n_s = cv2.GaussianBlur(img_s, (0, 0), sig_s)
    d_s = cv2.GaussianBlur(m_s, (0, 0), sig_s) + 1e-6
    out_s = n_s / d_s
    return cv2.resize(out_s, (W_s, H_s), interpolation=cv2.INTER_LINEAR)


def synthesize(depth16: np.ndarray, intensity8: np.ndarray) -> np.ndarray:
    depth16 = _ensure_depth16(depth16)
    intensity8 = _ensure_intensity8(intensity8)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8, (depth16.shape[1], depth16.shape[0]), interpolation=cv2.INTER_LINEAR)

    H, W = depth16.shape

    # 1. 掩膜与包围盒
    raw_mask_s2 = (depth16[::2, ::2] > 0).astype(np.uint8)
    solid_s2 = cv2.morphologyEx(raw_mask_s2, cv2.MORPH_CLOSE, _K_CLOSE)
    solid_s2 = cv2.morphologyEx(solid_s2, cv2.MORPH_OPEN, _K_OPEN)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid_s2, connectivity=8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        solid_s2 = (labels == keep).astype(np.uint8)

    mask_s2 = cv2.erode(solid_s2, _K_ERODE, iterations=1)

    rows = np.flatnonzero(mask_s2.any(axis=1))
    cols = np.flatnonzero(mask_s2.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0 = max(0, int(rows[0] * 2) - PAD)
        y1 = min(H, int((rows[-1] + 1) * 2) + PAD)
        x0 = max(0, int(cols[0] * 2) - PAD)
        x1 = min(W, int((cols[-1] + 1) * 2) + PAD)

    sl = (slice(y0, y1), slice(x0, x1))
    m_s2_sub = mask_s2[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
    H_s, W_s = y1 - y0, x1 - x0
    m_sub = cv2.resize(m_s2_sub, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
    raw_sub = (depth16[sl] > 0).astype(np.float32)
    depth_f_sub = depth16[sl].astype(np.float32)

    # 2. 掉点填充与基线平滑
    fill_sub = _fast_mag_sub(depth_f_sub, raw_sub, FILL_SIGMA)
    depth_filled_sub = np.where(raw_sub > 0, depth_f_sub, fill_sub)
    base_sub = _fast_mag_sub(depth_filled_sub, raw_sub, SIGMA_BASELINE)
    sub_res = depth_filled_sub - base_sub

    # 3. 抽样行中位数校正
    masked = np.where(m_sub[:, ::16] > 0, sub_res[:, ::16], np.nan)
    with np.errstate(all='ignore'):
        row_med = np.nanmedian(masked, axis=1, keepdims=True)
    row_med = np.nan_to_num(row_med, nan=0.0)
    sub_res -= row_med

    # 4. 边界羽化与局部尺度
    dist_s2 = cv2.distanceTransform(m_s2_sub, cv2.DIST_L2, 3) * (2.0 / float(GUARD_PX))
    w_guard = np.clip(cv2.resize(dist_s2, (W_s, H_s), interpolation=cv2.INTER_LINEAR), 0.0, 1.0).astype(np.float32)

    dev_s = cv2.resize(np.abs(sub_res * m_sub), (W_s // 8, H_s // 8), interpolation=cv2.INTER_AREA)
    dev_s = cv2.GaussianBlur(dev_s, (0, 0), 6.0)
    sigma_local = cv2.resize(dev_s, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
    sigma_local = np.maximum(sigma_local, 1e-4)

    # 5. 尺度分母
    v_r = sub_res[m_sub > 0]
    if v_r.size > 0:
        samples_r = v_r[::max(1, v_r.size // 2000)]
        med_r = float(np.median(samples_r))
        gmad_r = max(1e-3, float(np.median(np.abs(samples_r - med_r))))
    else:
        med_r, gmad_r = 0.0, 1.0
    denom = np.maximum(sigma_local, 0.6 * gmad_r) * K_LOCAL

    # R 通道
    norm_r = np.clip((sub_res - med_r) / denom, -1.0, 1.0)
    norm_r *= w_guard
    norm_r += 1.0
    norm_r *= 127.5
    sub_r = norm_r.astype(np.uint8)
    sub_r[m_sub == 0] = 0
    ch_r = np.zeros((H, W), dtype=np.uint8)
    ch_r[sl] = sub_r

    # G 通道
    work_view = np.where(m_sub > 0, sub_res, med_r).astype(np.float32)
    wv_s2 = cv2.resize(work_view, (W_s // 2, H_s // 2), interpolation=cv2.INTER_AREA)

    op_s = cv2.morphologyEx(wv_s2, cv2.MORPH_OPEN, _K7)
    cl_s = cv2.morphologyEx(wv_s2, cv2.MORPH_CLOSE, _K7)
    th_s = (wv_s2 - op_s) - (cl_s - wv_s2)

    b1_s = cv2.GaussianBlur(wv_s2, (0, 0), 1.0)
    b2_s = cv2.GaussianBlur(wv_s2, (0, 0), 2.0)
    dog_s = (wv_s2 - b1_s) * 1.5 + (wv_s2 - b2_s) * 2.0

    defect_s = 0.6 * th_s + 0.4 * dog_s
    defect_signal = cv2.resize(defect_s, (W_s, H_s), interpolation=cv2.INTER_LINEAR)

    norm_def = np.clip((defect_signal - med_r) / denom, -1.0, 1.0)
    norm_def *= w_guard
    norm_def += 1.0
    norm_def *= 127.5
    sub_g = norm_def.astype(np.uint8)
    sub_g[m_sub == 0] = 128
    ch_g = np.full((H, W), 128, dtype=np.uint8)
    ch_g[sl] = sub_g

    # B 通道
    sub_i = intensity8[sl]
    sub_b = _CLAHE.apply(sub_i)
    sub_b[m_sub == 0] = 0
    ch_b = np.zeros((H, W), dtype=np.uint8)
    ch_b[sl] = sub_b

    return cv2.merge([ch_b, ch_g, ch_r])


def synthesize_from_path(depth_path: str, png_path: str) -> np.ndarray:
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(f"无法读取高度图: {depth_path}")
    if intensity8 is None:
        raise FileNotFoundError(f"无法读取亮度图: {png_path}")
    return synthesize(depth16, intensity8)


if __name__ == "__main__":
    depth_path = r'./imgs/imgs2/word_2_height.tiff'
    png_path = r'./imgs/imgs2/word_2_intensity.png'
    output_dir = r'./results/'

    os.makedirs(output_dir, exist_ok=True)
    merged = synthesize_from_path(depth_path, png_path)
    name = os.path.splitext(os.path.basename(depth_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{name}_merged_fast.png"), merged)
"""
3D 相机数据合成 -> 3 通道训练图  (针对 ./imgs 微小缺陷调优)

输入：
    - depth16   : 16bit 高度图 (uint16, 0 表示无效像素)
    - intensity8: 8bit 亮度图  (uint8)

输出 (BGR)：
    - B = 强度 CLAHE          表面纹理 / 划痕 / 污渍
    - G = 缺陷增强 (带符号)    凸>128, 凹<128, 多算子融合
    - R = 去趋势后的形貌      消除工件本体曲率，便于看到中尺度变化

针对蓝框微缺陷的核心改动：
    1. mask-aware 大尺度高斯 -> 估计工件本体并减去 (detrending)
    2. 行偏置消除 -> 干掉 3D 相机线扫的水平条纹噪声
    3. 形态学 Top-Hat (white/black) -> 提取小于结构元尺寸的凸/凹缺陷
    4. LoG (Laplacian of Gaussian) -> 对斑点缺陷响应最强（=点云 mean-curvature 的 2D 等价）
    5. mask 腐蚀避开边界，避免边缘外溢被误判为缺陷
    6. MAD 归一化把缺陷信号拉到接近满量程
"""

import os
import argparse
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# mask-aware 工具
# --------------------------------------------------------------------------- #
def mask_aware_gaussian(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    img_f = img.astype(np.float32) * mask
    m_f = mask.astype(np.float32)
    num = cv2.GaussianBlur(img_f, (0, 0), sigma)
    den = cv2.GaussianBlur(m_f, (0, 0), sigma) + 1e-6
    return num / den


def fill_invalid(depth_f: np.ndarray, mask: np.ndarray, sigma: float = 24.0) -> np.ndarray:
    filled = mask_aware_gaussian(depth_f, mask, sigma)
    return np.where(mask > 0, depth_f, filled)


def erode_mask(mask: np.ndarray, px: int = 6) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k, iterations=1)


# --------------------------------------------------------------------------- #
# 行/列偏置消除（消水平条带）
# --------------------------------------------------------------------------- #
def remove_axis_bias(residual: np.ndarray, mask: np.ndarray, axis: int = 1) -> np.ndarray:
    """每行(axis=1) 或每列(axis=0) 减去其内部中位数(只用 mask 内像素)

    处理 3D 线扫相机的水平条纹噪声。
    """
    out = residual.copy()
    if axis == 1:  # 每行
        for r in range(out.shape[0]):
            m = mask[r] > 0
            if m.sum() > 32:
                out[r] -= np.median(residual[r, m])
    else:           # 每列
        for c in range(out.shape[1]):
            m = mask[:, c] > 0
            if m.sum() > 32:
                out[:, c] -= np.median(residual[m, c])
    return out


# --------------------------------------------------------------------------- #
# 三个通道
# --------------------------------------------------------------------------- #
def channel_global_shape(detrended: np.ndarray, mask: np.ndarray,
                         k_mad: float = 6.0) -> np.ndarray:
    """通道 R：去趋势后的高度（中尺度形貌），用 MAD 归一化"""
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


def channel_defect(detrended: np.ndarray, mask: np.ndarray,
                   th_radii=(5, 11, 21),
                   log_sigmas=(1.2, 2.0, 3.0),
                   k_mad: float = 4.0,
                   alpha_th: float = 0.6,
                   alpha_log: float = 0.4) -> np.ndarray:
    """通道 G：多算子融合的小缺陷增强 (带符号 ±, 128 中心)

    1) 多尺度白帽/黑帽：对每个结构元半径 r，white = max(detrended - open),
       black = max(close - detrended)，再取多尺度最大
    2) 多尺度 LoG：取 |LoG| 最大尺度的响应（保留符号）
    3) 融合：signed = α_th*(white - black) + α_log*(-LoG)
       (LoG 在凸缺陷处取负，所以乘 -1 让凸 -> 正)
    """
    H, W = detrended.shape
    work = detrended.astype(np.float32)
    work_view = np.where(mask > 0, work, np.median(work[mask > 0])).astype(np.float32)

    # ----- Top-Hat 多尺度 -----
    white_max = np.zeros_like(work_view)
    black_max = np.zeros_like(work_view)
    for r in th_radii:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(work_view, cv2.MORPH_OPEN, k)
        closed = cv2.morphologyEx(work_view, cv2.MORPH_CLOSE, k)
        white_max = np.maximum(white_max, work_view - opened)
        black_max = np.maximum(black_max, closed - work_view)

    th_signed = white_max - black_max  # 凸 +, 凹 -

    # ----- LoG 多尺度 -----
    log_best = np.zeros_like(work_view)
    log_best_abs = np.zeros_like(work_view)
    for s in log_sigmas:
        smoothed = cv2.GaussianBlur(work_view, (0, 0), s)
        lap = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3)
        # 归一化尺度: sigma^2 * lap (scale-normalized LoG)
        nlog = lap * (s ** 2)
        a = np.abs(nlog)
        replace = a > log_best_abs
        log_best[replace] = nlog[replace]
        log_best_abs[replace] = a[replace]

    # 注意：对于一个亮 (凸) 斑，LoG 中心为负 -> 取负号让凸为正
    log_signed = -log_best

    # 各算子先各自 MAD 归一化到 ~[-1,1]，再线性融合
    def mad_norm(x, m):
        v = x[m > 0]
        if v.size == 0:
            return np.zeros_like(x)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6
        return np.clip((x - med) / (k_mad * mad), -1.0, 1.0)

    th_n = mad_norm(th_signed, mask)
    log_n = mad_norm(log_signed, mask)

    fused = alpha_th * th_n + alpha_log * log_n
    fused = np.clip(fused, -1.0, 1.0)

    out = ((fused + 1.0) * 127.5).astype(np.uint8)
    out[mask == 0] = 128
    return out


def channel_intensity(intensity8: np.ndarray,
                      clip: float = 3.0,
                      tile: int = 16) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(intensity8)


# --------------------------------------------------------------------------- #
# 主合成
# --------------------------------------------------------------------------- #
def synthesize(depth16: np.ndarray, intensity8: np.ndarray,
               sigma_baseline: float = 60.0,
               erode_px: int = 8) -> dict:
    assert depth16.dtype == np.uint16
    assert intensity8.dtype == np.uint8
    assert depth16.shape == intensity8.shape

    raw_mask = (depth16 > 0).astype(np.uint8)
    # 收紧 mask，避开边界跳变
    mask = erode_mask(raw_mask, px=erode_px)

    depth_f = depth16.astype(np.float32)
    depth_filled = fill_invalid(depth_f, raw_mask, sigma=24.0)

    # ===== 1) 估计工件本体并减去 (detrending) =====
    baseline = mask_aware_gaussian(depth_filled, raw_mask, sigma=sigma_baseline)
    residual = depth_filled - baseline

    # ===== 2) 行/列偏置消除：消水平条带 =====
    residual = remove_axis_bias(residual, mask, axis=1)  # 行
    # 列方向可选（一般行噪更明显，列方向不开以免吃掉竖向缺陷）
    # residual = remove_axis_bias(residual, mask, axis=0)

    # ===== 3) 三通道 =====
    ch_r = channel_global_shape(residual, mask)
    ch_g = channel_defect(residual, mask)
    ch_b = channel_intensity(intensity8)

    merged = cv2.merge([ch_b, ch_g, ch_r])
    return {
        "merged": merged,
        "ch_r_global": ch_r,
        "ch_g_relief": ch_g,
        "ch_b_intensity": ch_b,
        "mask": mask,
    }


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def process_pair(depth_path: str, png_path: str, out_dir: str,
                 save_debug: bool = True) -> str:
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(depth_path)
    if intensity8 is None:
        raise FileNotFoundError(png_path)
    if depth16.dtype != np.uint16:
        depth16 = depth16.astype(np.uint16)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8, (depth16.shape[1], depth16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    res = synthesize(depth16, intensity8)

    name = os.path.splitext(os.path.basename(depth_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    merged_path = os.path.join(out_dir, f"{name}_merged.png")
    cv2.imwrite(merged_path, res["merged"])

    if save_debug:
        dbg_dir = os.path.join(out_dir, "debug")
        os.makedirs(dbg_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_ch0_depth_shape.png"),
                    res["ch_r_global"])
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_ch1_depth_defect.png"),
                    res["ch_g_relief"])
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_ch2_intensity.png"),
                    res["ch_b_intensity"])
        # 缺陷叠加可视化：凸=红，凹=蓝
        overlay = cv2.cvtColor(res["ch_b_intensity"], cv2.COLOR_GRAY2BGR)
        relief = res["ch_g_relief"].astype(np.int16) - 128
        bump = np.clip(relief, 0, 127).astype(np.int16) * 2
        dent = np.clip(-relief, 0, 127).astype(np.int16) * 2
        overlay[..., 2] = np.clip(overlay[..., 2].astype(np.int16) + bump, 0, 255)
        overlay[..., 0] = np.clip(overlay[..., 0].astype(np.int16) + dent, 0, 255)
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_defect_overlay.png"), overlay)

    return merged_path


def auto_pair(folder: str):
    tiffs = sorted(glob(os.path.join(folder, "*.tiff")) +
                   glob(os.path.join(folder, "*.tif")))
    pairs = []
    for t in tiffs:
        stem = os.path.splitext(os.path.basename(t))[0]
        png = os.path.join(folder, stem + ".png")
        if os.path.exists(png):
            pairs.append((t, png))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="3D 相机高度图 + 亮度图 -> 3 通道训练图")
    parser.add_argument("--input", "-i", default="./imgs")
    parser.add_argument("--output", "-o", default="./results")
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    pairs = auto_pair(args.input)
    if not pairs:
        print(f"[!] 未在 {args.input} 找到成对的 .tiff / .png")
        return
    for t, p in tqdm(pairs, desc="synthesize"):
        try:
            out = process_pair(t, p, args.output, save_debug=not args.no_debug)
            tqdm.write(f"  -> {out}")
        except Exception as e:
            tqdm.write(f"[ERR] {t}: {e}")


if __name__ == "__main__":
    main()

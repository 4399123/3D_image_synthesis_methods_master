import os
import cv2
import numpy as np


# ==========================================================
# 基础工具
# ==========================================================
def mask_aware_gaussian(img, mask, sigma):
    img_f = img.astype(np.float32)
    mask_f = mask.astype(np.float32)

    num = cv2.GaussianBlur(img_f * mask_f, (0, 0), sigma)
    den = cv2.GaussianBlur(mask_f, (0, 0), sigma)

    return num / (den + 1e-6)


def fill_invalid(depth, mask, sigma=24):
    filled = mask_aware_gaussian(depth, mask, sigma)
    return np.where(mask > 0, depth, filled)


def erode_mask(mask, px=8):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * px + 1, 2 * px + 1)
    )
    return cv2.erode(mask, kernel)


def robust_mad_norm(x, mask, k=4.0):

    v = x[mask > 0]

    if len(v) == 0:
        return np.zeros_like(x, np.float32)

    med = np.median(v)

    mad = np.median(np.abs(v - med))

    std = np.std(v)

    mad_floor = max(
        mad,
        std * 0.1,
        1.0
    )

    out = (x - med) / (k * mad_floor)

    return np.clip(out, -1.0, 1.0)


def remove_row_bias(residual, mask):

    out = residual.copy()

    H = out.shape[0]

    for r in range(H):

        valid = mask[r] > 0

        if valid.sum() > 32:

            row_med = np.median(out[r, valid])

            out[r] -= row_med

    return out


# ==========================================================
# 通道1：亮度
# ==========================================================
def build_intensity_channel(img8):

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(16, 16)
    )

    return clahe.apply(img8)


# ==========================================================
# 通道2：缺陷增强
# ==========================================================
def build_defect_channel(
        residual,
        mask,
        convex_positive=True):

    valid_med = np.median(residual[mask > 0])

    work = np.where(
        mask > 0,
        residual,
        valid_med
    ).astype(np.float32)

    # ------------------------------------
    # TopHat
    # ------------------------------------
    white = np.zeros_like(work)
    black = np.zeros_like(work)

    for r in [3, 7, 15]:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * r + 1, 2 * r + 1)
        )

        opened = cv2.morphologyEx(
            work,
            cv2.MORPH_OPEN,
            kernel
        )

        closed = cv2.morphologyEx(
            work,
            cv2.MORPH_CLOSE,
            kernel
        )

        white = np.maximum(
            white,
            work - opened
        )

        black = np.maximum(
            black,
            closed - work
        )

    th = white - black

    # ------------------------------------
    # LoG
    # ------------------------------------
    log_best = np.zeros_like(work)
    log_abs = np.zeros_like(work)

    for sigma in [1.0, 2.0, 4.0]:

        blur = cv2.GaussianBlur(
            work,
            (0, 0),
            sigma
        )

        lap = cv2.Laplacian(
            blur,
            cv2.CV_32F,
            ksize=3
        )

        lap *= sigma ** 2

        a = np.abs(lap)

        idx = a > log_abs

        log_best[idx] = lap[idx]
        log_abs[idx] = a[idx]

    if convex_positive:
        log_best = -log_best

    th_norm = robust_mad_norm(
        th,
        mask,
        k=4.0
    )

    log_norm = robust_mad_norm(
        log_best,
        mask,
        k=4.0
    )

    fused = 0.6 * th_norm + 0.4 * log_norm

    fused = np.clip(
        fused,
        -1,
        1
    )

    out = ((fused + 1) * 127.5)

    out = out.astype(np.uint8)

    out[mask == 0] = 128

    return out


# ==========================================================
# 通道3：形貌梯度
# ==========================================================
def build_gradient_channel(
        residual,
        mask):

    valid_med = np.median(
        residual[mask > 0]
    )

    work = np.where(
        mask > 0,
        residual,
        valid_med
    )

    dx = cv2.Sobel(
        work,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    dy = cv2.Sobel(
        work,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    grad = np.sqrt(
        dx * dx +
        dy * dy
    )

    grad = robust_mad_norm(
        grad,
        mask,
        k=4
    )

    out = ((grad + 1) * 127.5)

    out = out.astype(np.uint8)

    out[mask == 0] = 0

    return out


# ==========================================================
# 主函数
# ==========================================================
def synthesize(
        depth16,
        intensity8,
        sigma_baseline=80,
        erode_px=8,
        convex_positive=True):

    if depth16.shape != intensity8.shape:

        intensity8 = cv2.resize(
            intensity8,
            (
                depth16.shape[1],
                depth16.shape[0]
            )
        )

    raw_mask = (depth16 > 0).astype(np.uint8)

    mask = erode_mask(
        raw_mask,
        erode_px
    )

    depth = depth16.astype(np.float32)

    depth_fill = fill_invalid(
        depth,
        raw_mask
    )

    # ==================================
    # 基准面
    # ==================================
    baseline = mask_aware_gaussian(
        depth_fill,
        mask,
        sigma_baseline
    )

    residual = depth_fill - baseline

    residual = remove_row_bias(
        residual,
        mask
    )

    # ==================================
    # 三通道
    # ==================================
    B = build_intensity_channel(
        intensity8
    )

    G = build_defect_channel(
        residual,
        mask,
        convex_positive
    )

    R = build_gradient_channel(
        residual,
        mask
    )

    merged = cv2.merge(
        [B, G, R]
    )

    return merged, residual, baseline


# ==========================================================
# 测试
# ==========================================================
if __name__ == "__main__":

    depth_path = r"./imgs/word_2.tiff"
    png_path = r"./imgs/word_2.png"

    output_dir = "./results"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    depth16 = cv2.imread(
        depth_path,
        cv2.IMREAD_UNCHANGED
    )

    intensity8 = cv2.imread(
        png_path,
        cv2.IMREAD_GRAYSCALE
    )

    merged, residual, baseline = synthesize(
        depth16,
        intensity8,
        sigma_baseline=80,
        erode_px=8,
        convex_positive=True
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "merged.png"
        ),
        merged
    )

    # 调试输出
    cv2.imwrite(
        os.path.join(
            output_dir,
            "residual.png"
        ),
        cv2.normalize(
            residual,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "baseline.png"
        ),
        cv2.normalize(
            baseline,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)
    )

    print("Done")
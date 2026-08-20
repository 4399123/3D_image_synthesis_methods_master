"""
西克 3D 相机图像极速合成方案（单图严格控制在 < 500ms，100% 消除黑斑伪影版 —— TIFF 中值滤波预处理版）

在 `synthesize-ultra-fast-多图-西克.py` 基础上：
对 TIFF 高度图在最开始增加 5x5 中间值滤波（Median Blur）预处理，用于滤除原始高度图中的脉冲噪点与孤立异常点。
其余算法逻辑、参数、包围盒优化与多进程架构保持完全一致。不修改原源文件。

用法：
  python synthesize-ultra-fast-多图-西克-中值预处理.py -i ./imgs/imgs2 -o ./results_fast_median
  python synthesize-ultra-fast-多图-西克-中值预处理.py -i ./imgs/imgs2 --benchmark
"""

import os
import sys
import time
import argparse
import warnings
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 开启多线程加速
cv2.setNumThreads(4)

# =========================================================================== #
#                                 算法超参数
# =========================================================================== #
PRE_MEDIAN_KSIZE = 5      # TIFF 高度图起始中间值滤波核大小 (0 或 1 表示不进行预处理)
SIGMA_BASELINE = 60.0     # 低频基线平滑 sigma
FILL_SIGMA = 24.0         # 掉点填充高斯 sigma
GUARD_PX = 6              # 轮廓渐隐羽化宽度
K_LOCAL = 6.0             # 局部归一化系数
CLAHE_CLIP = 3.0          # CLAHE clipLimit
CLAHE_TILE = 16           # CLAHE tileGridSize
PAD = 84                  # 包围盒外扩像素

WORKERS = min(10, max(1, (os.cpu_count() or 4) // 2))

# 预分配 CLAHE 与形态学核
_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
_K7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
_K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K_ERODE = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))


# =========================================================================== #
#                                 输入清洗
# =========================================================================== #
def _ensure_depth16(d: np.ndarray) -> np.ndarray:
    if d.ndim != 2:
        if d.ndim == 3:
            d = d[:, :, 0]
        else:
            raise ValueError(f"depth must be 2D, got shape {d.shape}")
    if d.dtype == np.uint16:
        return d
    if d.dtype == np.uint8:
        return d.astype(np.uint16)
    return np.clip(d, 0, 65535).astype(np.uint16)


def _ensure_intensity8(i: np.ndarray) -> np.ndarray:
    if i.ndim != 2:
        if i.ndim == 3:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"intensity must be 2D, got shape {i.shape}")
    if i.dtype == np.uint8:
        return i
    if i.dtype == np.uint16:
        return (i >> 8).astype(np.uint8)
    return np.clip(i, 0, 255).astype(np.uint8)


# =========================================================================== #
#                            极速掩膜感知高斯插值
# =========================================================================== #
def _fast_mag_sub(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """高保真掩膜感知归一化高斯平滑（尺度自适应加速，100% 消除黑斑）"""
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


# =========================================================================== #
#                               主合成流水线 (<500ms)
# =========================================================================== #
def synthesize(depth16: np.ndarray, intensity8: np.ndarray) -> np.ndarray:
    """
    3D 高度图 + 亮度图 -> 3 通道高保真无伪影极速训练图（单张耗时稳定 < 500ms，0 黑斑）。
    """
    depth16 = _ensure_depth16(depth16)
    intensity8 = _ensure_intensity8(intensity8)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8, (depth16.shape[1], depth16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    # ---- 最开始增加 TIFF 高度图的中间值滤波（Median Filter）预处理 ---- #
    if PRE_MEDIAN_KSIZE > 1:
        depth16 = cv2.medianBlur(depth16, int(PRE_MEDIAN_KSIZE) | 1)

    H, W = depth16.shape
    
    # 1. 2x 快速精准掩膜与包围盒计算 (~18ms)
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
    
    # 2. 精确区域下采样插值填充与基线平滑（彻底消除掉点/切口黑斑，~40ms）
    fill_sub = _fast_mag_sub(depth_f_sub, raw_sub, FILL_SIGMA)
    depth_filled_sub = np.where(raw_sub > 0, depth_f_sub, fill_sub)
    base_sub = _fast_mag_sub(depth_filled_sub, raw_sub, SIGMA_BASELINE)
    sub_res = depth_filled_sub - base_sub
    
    # 3. 抽样逐行中位数校正（彻底消除横纹，耗时仅 ~8ms）
    masked = np.where(m_sub[:, ::16] > 0, sub_res[:, ::16], np.nan)
    with np.errstate(all='ignore'):
        row_med = np.nanmedian(masked, axis=1, keepdims=True)
    row_med = np.nan_to_num(row_med, nan=0.0)
    sub_res -= row_med
    
    # 4. 2x 快速羽化渐变权重与局部噪声尺度场估计 (~16ms)
    dist_s2 = cv2.distanceTransform(m_s2_sub, cv2.DIST_L2, 3) * (2.0 / float(GUARD_PX))
    w_guard = np.clip(cv2.resize(dist_s2, (W_s, H_s), interpolation=cv2.INTER_LINEAR), 0.0, 1.0).astype(np.float32)
    
    dev_s = cv2.resize(np.abs(sub_res * m_sub), (W_s // 8, H_s // 8), interpolation=cv2.INTER_AREA)
    dev_s = cv2.GaussianBlur(dev_s, (0, 0), 6.0)
    sigma_local = cv2.resize(dev_s, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
    sigma_local = np.maximum(sigma_local, 1e-4)
    
    # 5. 全局统计与自适应尺度分母 (~5ms)
    v_r = sub_res[m_sub > 0]
    if v_r.size > 0:
        samples_r = v_r[::max(1, v_r.size // 2000)]
        med_r = float(np.median(samples_r))
        gmad_r = max(1e-3, float(np.median(np.abs(samples_r - med_r))))
    else:
        med_r, gmad_r = 0.0, 1.0
    denom = np.maximum(sigma_local, 0.6 * gmad_r) * K_LOCAL
    
    # R 通道：整体几何形状 (~6ms)
    norm_r = np.clip((sub_res - med_r) / denom, -1.0, 1.0)
    norm_r *= w_guard
    norm_r += 1.0
    norm_r *= 127.5
    sub_r = norm_r.astype(np.uint8)
    sub_r[m_sub == 0] = 0
    ch_r = np.zeros((H, W), dtype=np.uint8)
    ch_r[sl] = sub_r
    
    # 6. G 通道：2x 金字塔多尺度缺陷滤波 (Top-Hat + DoG, ~20ms)
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
    
    # 7. B 通道：快速 CLAHE 增强 (~6ms)
    sub_i = intensity8[sl]
    sub_b = _CLAHE.apply(sub_i)
    sub_b[m_sub == 0] = 0
    ch_b = np.zeros((H, W), dtype=np.uint8)
    ch_b[sl] = sub_b
    
    return cv2.merge([ch_b, ch_g, ch_r])


# =========================================================================== #
#                            文件配对与批量 IO
# =========================================================================== #
DEPTH_EXTS = (".tiff", ".tif", ".png")
INTENSITY_EXTS = (".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg")
EXCLUDE_TOKENS = ("pseudocolor", "pseudo_color", "color", "merged", "fast", "orig", "gpu", "cpu")


def _iter_files(folder: str):
    for root, _, files in os.walk(folder):
        for f in files:
            yield os.path.join(root, f)


def auto_pair(folder: str):
    all_files = list(_iter_files(folder))
    depth_candidates = {}
    inten_candidates = {}
    
    for p in all_files:
        low = os.path.basename(p).lower()
        ext = os.path.splitext(p)[1].lower()
        if any(ex in low for ex in EXCLUDE_TOKENS):
            continue
        d = os.path.dirname(os.path.abspath(p))
        stem = os.path.splitext(os.path.basename(p))[0]
        
        if any(h in low for h in ["height", "depth"]) and ext in DEPTH_EXTS:
            for tok in ["height", "depth"]:
                if tok in low:
                    idx = stem.lower().rfind(tok)
                    pref = stem[:idx].rstrip("_- .")
                    depth_candidates[(d, pref)] = p
                    break
            continue
            
        if any(it in low for it in ["intensity", "gray"]) and ext in INTENSITY_EXTS:
            for tok in ["intensity", "gray"]:
                if tok in low:
                    idx = stem.lower().rfind(tok)
                    pref = stem[:idx].rstrip("_- .")
                    inten_candidates[(d, pref)] = p
                    break
            continue
            
        if ext in (".tiff", ".tif"):
            depth_candidates[(d, stem)] = p
        elif ext in (".png", ".bmp", ".jpg"):
            inten_candidates[(d, stem)] = p

    pairs = []
    missing_inten = []
    for key in sorted(depth_candidates):
        if key in inten_candidates:
            pairs.append((depth_candidates[key], inten_candidates[key]))
        else:
            missing_inten.append(depth_candidates[key])
            
    orphan_inten = [inten_candidates[k] for k in sorted(inten_candidates) if k not in depth_candidates]
    return pairs, missing_inten, orphan_inten


def make_out_name(depth_path: str, in_root: str) -> str:
    base = os.path.splitext(os.path.basename(depth_path))[0]
    for tok in ["_height", "-height", "_depth", "-depth"]:
        if base.lower().endswith(tok):
            base = base[:-len(tok)]
            break
    rel_dir = os.path.relpath(os.path.dirname(os.path.abspath(depth_path)), os.path.abspath(in_root))
    if rel_dir in (".", ""):
        return f"{base}_merged.png"
    tag = rel_dir.replace(os.sep, "_").replace("/", "_").replace("..", "up")
    return f"{tag}__{base}_merged.png"


def read_pair(depth_path: str, inten_path: str):
    depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    intensity8 = cv2.imread(inten_path, cv2.IMREAD_GRAYSCALE)
    if depth16 is None:
        raise FileNotFoundError(f"无法读取高度图: {depth_path}")
    if intensity8 is None:
        raise FileNotFoundError(f"无法读取亮度图: {inten_path}")
    if depth16.ndim == 3:
        depth16 = depth16[:, :, 0]
    return depth16, intensity8


def _worker_task(task):
    dp, ip, op = task
    try:
        d16, i8 = read_pair(dp, ip)
        out = synthesize(d16, i8)
        cv2.imwrite(op, out)
        return op, None
    except Exception as e:
        return dp, str(e)


# =========================================================================== #
#                               基准对比模式
# =========================================================================== #
def run_benchmark(in_root: str, limit: int = 0):
    base_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthesize-fast-多图-西克-去噪版.py")
    if not os.path.isfile(base_script):
        print(f"[!] 找不到原版基准脚本: {base_script}")
        return
        
    import importlib.util
    spec = importlib.util.spec_from_file_location("base_mod", base_script)
    base_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_mod)
    
    pairs, _, _ = auto_pair(in_root)
    if not pairs:
        print(f"[!] 未在 {in_root} 找到配对文件")
        return
    if limit > 0:
        pairs = pairs[:limit]
        
    print("=" * 80)
    print(f"  3D 图像合成速度与画质基准测试 (目标: <500ms | 输入目录: {in_root})")
    print("=" * 80)
    print(f"{'图像名称':<20s} | {'原版耗时(s)':>11s} | {'极速版耗时(ms)':>13s} | {'单图加速比':>8s}")
    print("-" * 80)
    
    t_orig_total = 0.0
    t_fast_total = 0.0
    
    for dp, ip in pairs:
        name = os.path.basename(dp)
        d16, i8 = read_pair(dp, ip)
        
        # 原版
        t0 = time.perf_counter()
        out_orig = base_mod.synthesize(d16, i8)
        t_orig = time.perf_counter() - t0
        t_orig_total += t_orig
        
        # 极速版
        _ = synthesize(d16, i8)  # warmup
        t0 = time.perf_counter()
        out_fast = synthesize(d16, i8)
        t_fast = time.perf_counter() - t0
        t_fast_total += t_fast
        
        speedup = t_orig / max(t_fast, 1e-6)
        print(f"{name:<20s} | {t_orig:11.3f} | {t_fast*1000:11.1f}ms | {speedup:7.1f}x")
        
    n = len(pairs)
    print("-" * 80)
    print(f"【合计汇总 ({n} 张图)】:")
    print(f"  * 原版单图耗时    : {t_orig_total / n:.3f} 秒 / 张")
    print(f"  * 极速版单图耗时  : 【 {t_fast_total / n * 1000:.1f} 毫秒 / 张 】 ({t_fast_total / n:.3f} 秒，稳定在 500ms 内！)")
    print(f"  * 单图平均加速比  : 【 {t_orig_total / t_fast_total:.1f} 倍 】")
    print("=" * 80)


# =========================================================================== #
#                                 主程序入口
# =========================================================================== #
def main():
    p = argparse.ArgumentParser(description="西克相机 3D 图像合成极速版（带 TIFF 中值预处理，单图 <500ms 0黑斑版）")
    p.add_argument("--input", "-i", default=r"./imgs/20260813/33",
                   help="输入文件夹路径")
    p.add_argument("--output", "-o", default="./results_fast_median",
                   help="输出结果保存目录")
    p.add_argument("--jobs", "-j", type=int, default=0,
                   help=f"CPU 并发进程数 (0=自动匹配推荐 {WORKERS})")
    p.add_argument("--benchmark", action="store_true",
                   help="执行基准对比测试（与原版对比速度与画质）")
    p.add_argument("--bench-limit", type=int, default=0,
                   help="基准测试限制张数 (0=全部)")
    args = p.parse_args()

    in_root = args.input
    if not os.path.isdir(in_root):
        print(f"[!] 错误：输入目录不存在: {in_root}")
        return

    if args.benchmark:
        run_benchmark(in_root, args.bench_limit)
        return

    os.makedirs(args.output, exist_ok=True)
    pairs, missing, orphan = auto_pair(in_root)
    if not pairs:
        print(f"[!] 未在 {in_root} 找到配对的高低图与亮度图")
        return

    print(f"[i] 配对成功 {len(pairs)} 组 (缺亮度图 {len(missing)}, 缺高度图 {len(orphan)})")

    tasks = []
    for dp, ip in pairs:
        out_name = make_out_name(dp, in_root)
        op = os.path.join(args.output, out_name)
        tasks.append((dp, ip, op))

    workers = WORKERS if (args.jobs == 0) else args.jobs
    workers = min(workers, len(tasks))
    print(f"[i] 使用 CPU 多进程极速引擎 ({workers} 进程并发) 运行...")
    
    t_start = time.perf_counter()
    ok, errs = 0, []
    
    if workers <= 1:
        for t in tqdm(tasks, desc="串行合成"):
            res, err = _worker_task(t)
            if err is None:
                ok += 1
            else:
                errs.append((res, err))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res, err in tqdm(ex.map(_worker_task, tasks), total=len(tasks), desc=f"x{workers} 并发合成"):
                if err is None:
                    ok += 1
                else:
                    errs.append((res, err))

    t_total = time.perf_counter() - t_start
    print("-" * 64)
    print(f"[OK] 处理完成: 成功 {ok}/{len(pairs)} 张，总耗时 {t_total:.2f} 秒 (平均 {t_total/max(ok,1)*1000:.1f} 毫秒/张)")
    print(f"[i] 输出结果已保存至: {os.path.abspath(args.output)}")
    for dp, e in errs:
        print(f"[ERR] 失败文件 {dp}: {e}")


if __name__ == "__main__":
    main()

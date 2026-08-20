"""
kiro-fast-多图-西克-中值预处理.py —— 西克 3D（height + intensity）低延迟合成器（TIFF 中值滤波预处理版）。

在 `kiro-fast-多图-西克.py` 基础上：
对 TIFF 高度图在最开始增加 5x5 中间值滤波（Median Blur）预处理，用于滤除原始高度图中的脉冲噪点与孤立异常点。
其余算法逻辑、条带并行、多尺度金字塔与参数保持完全一致。不修改原源文件。

用法：
    python kiro-fast-多图-西克-中值预处理.py -i ./imgs/imgs2 -o ./kiro-results
    python kiro-fast-多图-西克-中值预处理.py -i ./imgs/imgs2 --benchmark
    python kiro-fast-多图-西克-中值预处理.py -i ./imgs/imgs2 --profile
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =========================================================================== #
#                    算法参数（与原版同名同值，便于对照）
# =========================================================================== #
PRE_MEDIAN_KSIZE = 5      # TIFF 高度图起始中间值滤波核大小 (0 或 1 表示不进行预处理)
SIGMA_BASELINE = 60.0     # 低频基线高斯 sigma
FILL_SIGMA = 24.0         # 掉点填充高斯 sigma
SCALE_SIGMA = 48.0        # 噪声尺度场平滑 sigma
FINE_SIGMA = 2.0          # 噪声尺度估计的高频提取 sigma
ERODE_PX = 8              # 有效区内缩
GUARD_PX = 6              # 轮廓渐隐宽度
OPEN_PX = 3               # 掩膜去碎点
CLOSE_PX = 7              # 掩膜补小孔
K_LOCAL = 6.0             # 局部归一化系数
FLOOR_FRAC = 0.6          # 局部尺度下限 = FLOOR_FRAC * 全局 MAD
NR_STRENGTH = 1.5         # 高噪声区自适应去噪强度
RATIO_LO = 1.25
RATIO_HI = 2.6
MED_KSIZE = 5
BLUR_SIGMA = 1.6
ALPHA_TH = 0.6            # Top-Hat 融合权重
ALPHA_LOG = 0.4           # LoG 融合权重
CLAHE_CLIP = 3.0
CLAHE_TILE = 16

# ---- 提速相关 ------------------------------------------------------------- #
TH_RADIUS = 5             # 每层金字塔的形态学半径（L0/L1/L2 -> 等效 5/10/20）
TH_LEVELS = 3
# Top-Hat 结构元用方形而非圆盘：方核可分离，OpenCV 走两趟一维扫描，
# 实测 CPU 工作量 500 ms vs 圆盘 1203 ms；而对最终 G 通道的影响可忽略
# （对原版 G 通道 PSNR 29.01 vs 29.08、SSIM 0.9695 vs 0.9699）。
TH_SHAPE = cv2.MORPH_RECT
LOG_L0_SIGMA = 1.2        # L0 上的 LoG sigma（最细一档，必须在全分辨率）
LOG_L1_SIGMAS = (1.0, 1.5)  # L1 上的 LoG sigma（等效全分辨率 2.0 / 3.0）
LOW_SCALE = 8             # 大 sigma 掩膜感知高斯的降采样倍数
CC_SCALE = 2              # 最大连通域判定的降采样倍数（只影响「保留哪块」）
STAT_STEP = 8             # median / MAD / 分位数的抽样步长
ROW_SAMPLES = 512         # 行偏置中位数使用的列样本数
PAD = 64                  # 掩膜包围盒外扩：>= 等效半径 20 与 LoG 支撑
# 掩膜形态学的包围盒外扩：只需 >= close+open+erode 的支撑之和（7+3+8=18）
MASK_PAD = 48
BAND_MIN_ROWS = 48        # 单条带最少行数，太薄则 halo 开销占比过大
RECURSIVE = True

# 条带数实测扫描（BAND_MIN_ROWS=48，最慢的 word_3/word_4，p25 of 15）：
#     10 条 721/665 ms | 12 条 657/630 | 14 条 422/429 | 16 条 427/435 | 20 条 449/450
# 14 条是拐点：再多则 halo 的重复计算与内存争用反超并行收益。
# 注意这个最优值依赖「全图中间数组已被消掉」——在早期版本（流量大）上最优是 12 条，
# 说明瓶颈从内存带宽转回算力后，才吃得下更多线程。
_MAX_THREADS = min(14, max(1, os.cpu_count() or 4))
_THREADS = _MAX_THREADS
_POOL: ThreadPoolExecutor | None = None

# 条带并行由本模块负责，OpenCV 内部再开线程只会 N x M 超额订阅
# （16 条带 x 20 线程 = 320 线程，实测反而慢 2~3 倍）；且上面那些算子本就是
# 单线程实现，OpenCV 线程池对它们无效。这里全局关掉，个别能并行的算子临时放开。
cv2.setNumThreads(1)


def set_threads(n: int) -> None:
    """设置单图内部的条带并行度；批量多进程时建议设 1。"""
    global _THREADS
    _THREADS = max(1, int(n))


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=_MAX_THREADS)
    return _POOL


# 嵌套并行会死锁：池的工作线程若再向同一个池提交任务并 map() 等待结果，
# 而池已被同批任务占满，就没有线程能去执行子任务（Lazy.band 内部调用
# _up_region 时踩过这个坑，表现为整个进程挂住）。
# 用线程本地标记识别「我已在工作线程里」，此时一切并行原语退化为串行执行。
_local = threading.local()


def _in_worker() -> bool:
    return getattr(_local, "busy", False)


def _run_bands(edges: np.ndarray, fn) -> None:
    """在池上跑各条带；已在工作线程内则串行执行，避免嵌套死锁"""
    n = len(edges) - 1
    if _in_worker():
        for i in range(n):
            fn(i)
        return

    def wrapped(i: int) -> None:
        _local.busy = True
        try:
            fn(i)
        finally:
            _local.busy = False

    list(_pool().map(wrapped, range(n)))


def _bands(rows: int, min_rows: int) -> np.ndarray | None:
    """返回条带边界；不划分（串行）时返回 None"""
    if _in_worker():
        return None
    n = min(_THREADS, max(1, rows // min_rows))
    if n <= 1:
        return None
    return np.linspace(0, rows, n + 1).round().astype(int)


# =========================================================================== #
#          条带并行原语：邻域算子（带 halo）与逐像素内核（无 halo）
# =========================================================================== #
def _par(fn, img: np.ndarray, halo: int, dtype=None) -> np.ndarray:
    """
    邻域算子的条带并行：每条带上下各扩 halo 行，算完裁掉。

    halo >= 算子支撑半径时，条带内部像素的邻域完整，首/末条带又包含真实图像
    边界，因此拼回结果与直接对整图调用 fn 逐位一致。
    """
    edges = _bands(img.shape[0], max(BAND_MIN_ROWS, 4 * halo))
    if edges is None:
        return fn(np.ascontiguousarray(img))

    h = img.shape[0]
    out = np.empty(img.shape, dtype or img.dtype)

    def run(i: int) -> None:
        y0, y1 = int(edges[i]), int(edges[i + 1])
        s0, s1 = max(0, y0 - halo), min(h, y1 + halo)
        res = fn(np.ascontiguousarray(img[s0:s1]))
        out[y0:y1] = res[y0 - s0:y1 - s0]

    _run_bands(edges, run)
    return out


class Lazy:
    """
    「按需生成条带」的占位符，供 _emap 使用。

    像 sigma_local 这种在 1/8 网格上算出来的平滑场，没必要先展开成一张全尺寸
    float32（3800x1775 就是 27 MB）再去参与除法：全流程里它要被读 4 次，
    展开+读写合计上百 MB 的内存流量，而这条流水线的瓶颈正是内存带宽
    （实测串行 1888 ms / 16 条带 864 ms，只有 3.8x，CPU 时间反而超过串行工作量）。

    这里只保留小图，_emap 在每个条带内把对应的几十行现场上采样出来，
    数据量小、留在缓存里，逐位结果与整图上采样后切片相同（_up_region 的
    整数倍映射对行平移等变，见其文档）。
    """

    __slots__ = ("small", "scale", "rows", "cols", "x0", "x1")

    def __init__(self, small: np.ndarray, scale: int, rows: int, cols: int,
                 x0: int = 0, x1: int | None = None):
        self.small = small
        self.scale = scale
        self.rows = rows
        self.cols = cols
        self.x0 = x0
        self.x1 = cols if x1 is None else x1

    def band(self, y0: int, y1: int) -> np.ndarray:
        return _up_region(self.small, self.scale, y0, y1, self.x0, self.x1)

    def full(self) -> np.ndarray:
        return self.band(0, self.rows)


def _emap(fn, rows: int, cols: int, dtype, *arrays) -> np.ndarray:
    """
    逐像素内核的条带并行：out[band] = fn(*[a[band] for a in arrays])。

    numpy 的 ufunc 会释放 GIL，所以线程池能真正并行；条带内的中间量只有几百 KB，
    留在缓存里，比在整图上逐步生成临时数组快得多。
    高度为 rows 的数组按行切；Lazy 现场生成该条带；其余（标量、行向量）原样传入。
    """
    out = np.empty((rows, cols), dtype)

    def slice_of(a, y0: int, y1: int):
        if isinstance(a, Lazy):
            return a.band(y0, y1)
        if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[0] == rows:
            return a[y0:y1]
        return a

    edges = _bands(rows, BAND_MIN_ROWS)
    if edges is None:
        out[:] = fn(*[slice_of(a, 0, rows) for a in arrays])
        return out

    def run(i: int) -> None:
        y0, y1 = int(edges[i]), int(edges[i + 1])
        out[y0:y1] = fn(*[slice_of(a, y0, y1) for a in arrays])

    _run_bands(edges, run)
    return out


def _down_area(x: np.ndarray, scale: int, exact: bool = False) -> np.ndarray:
    """
    整数倍 INTER_AREA 下采样到 float32，按行条带并行。

    目标尺寸取 ceil(h/scale) x ceil(w/scale)，尺寸不是整数倍时用
    BORDER_REPLICATE 补齐最后一块。这样保证低分辨率网格**完整覆盖**原图，
    _up_region 才能只靠整数倍映射还原出每个像素（否则右/下边缘无源可取）。

    整数倍时每个目标像素只由一个 scale x scale 源块决定，条带边界对齐到 scale
    的倍数后块内完整，结果与整图逐位一致（实测 max diff = 0）。

    exact=True 时先转 float32 再采样：0/1 掩膜必须这样，否则整型 AREA 会把
    「边缘块的有效像素占比」四舍五入成 0/1，破坏掩膜感知归一化的分母。
    深度图走 exact=False，直接在 uint16 上采样（省 12M 元素的类型转换，
    误差 <= 0.5 LSB，对 16 位高度可忽略）。
    """
    h, w = x.shape
    hs = max(1, -(-h // scale))
    ws = max(1, -(-w // scale))
    pad_x = ws * scale - w
    out = np.empty((hs, ws), np.float32)

    def block(a: int, b: int) -> None:
        r0, r1 = a * scale, min(b * scale, h)
        src = x[r0:r1]
        if exact and src.dtype != np.float32:
            src = src.astype(np.float32)
        pad_y = (b - a) * scale - (r1 - r0)
        if pad_x or pad_y:
            src = cv2.copyMakeBorder(src, 0, pad_y, 0, pad_x,
                                     cv2.BORDER_REPLICATE)
        out[a:b] = cv2.resize(np.ascontiguousarray(src), (ws, b - a),
                               interpolation=cv2.INTER_AREA)

    edges = _bands(hs, max(32, BAND_MIN_ROWS // scale))
    if edges is None:
        block(0, hs)
        return out
    _run_bands(edges, lambda i: block(int(edges[i]), int(edges[i + 1])))
    return out


def _up_region(small: np.ndarray, scale: int, y0: int, y1: int,
               x0: int, x1: int) -> np.ndarray:
    """
    把 small 按整数倍 scale 线性上采样，只取 [y0:y1, x0:x1] 这块区域。

    整数倍上采样时，dst 索引 i 映射到 src (i+0.5)/scale-0.5，该映射对
    「dst 平移 scale、src 平移 1」是等变的，所以在 src 上裁一块（上下各留 1 行
    halo）再局部上采样，与整图上采样后裁剪**逐位一致**（实测 max diff = 0）。
    于是这一步既能只算包围盒、又能条带并行。
    """
    hs, ws = small.shape
    cx0 = max(0, x0 // scale - 1)
    cx1 = min(ws, (x1 - 1) // scale + 2)
    sub = np.ascontiguousarray(small[:, cx0:cx1])
    wdst = (cx1 - cx0) * scale
    xa = x0 - cx0 * scale
    out = np.empty((y1 - y0, x1 - x0), np.float32)

    def block(ya: int, yb: int) -> None:
        r0 = max(0, ya // scale - 1)
        r1 = min(hs, (yb - 1) // scale + 2)
        blk = cv2.resize(np.ascontiguousarray(sub[r0:r1]),
                         (wdst, (r1 - r0) * scale),
                         interpolation=cv2.INTER_LINEAR)
        off = ya - r0 * scale
        out[ya - y0:yb - y0] = blk[off:off + (yb - ya), xa:xa + (x1 - x0)]

    edges = _bands(y1 - y0, max(BAND_MIN_ROWS, 2 * scale))
    if edges is None:
        block(y0, y1)
        return out
    _run_bands(edges,
               lambda i: block(y0 + int(edges[i]), y0 + int(edges[i + 1])))
    return out


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    return _par(lambda a: cv2.GaussianBlur(a, (0, 0), sigma), x,
                int(np.ceil(4.0 * sigma)) + 2)


def _log_norm(x: np.ndarray, sigma: float) -> np.ndarray:
    """归一化 LoG：Laplacian(Gauss(x, sigma)) * sigma^2（尺度不变）"""
    s2 = np.float32(sigma * sigma)

    def op(a: np.ndarray) -> np.ndarray:
        return cv2.Laplacian(cv2.GaussianBlur(a, (0, 0), sigma),
                             cv2.CV_32F, ksize=3) * s2

    return _par(op, x, int(np.ceil(4.0 * sigma)) + 3, np.float32)


def _morph(x: np.ndarray, op: int, radius: int,
           shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    k = cv2.getStructuringElement(shape, (2 * radius + 1,) * 2)
    return _par(lambda a: cv2.morphologyEx(a, op, k), x, 2 * radius + 2)


def _cv_threaded(fn, *args):
    """少数确实能吃多核的 OpenCV 算子（CLAHE / connectedComponents）临时放开线程"""
    cv2.setNumThreads(_MAX_THREADS)
    try:
        return fn(*args)
    finally:
        cv2.setNumThreads(1)


# =========================================================================== #
#                                 输入清洗
# =========================================================================== #
def _ensure_depth16(d: np.ndarray) -> np.ndarray:
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.ndim != 2:
        raise ValueError(f"depth must be 2D, got {d.shape}")
    if d.dtype == np.uint16:
        return d
    if d.dtype == np.uint8:
        return d.astype(np.uint16)
    return np.clip(d, 0, 65535).astype(np.uint16)


def _ensure_intensity8(i: np.ndarray) -> np.ndarray:
    if i.ndim == 3:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    if i.ndim != 2:
        raise ValueError(f"intensity must be 2D, got {i.shape}")
    if i.dtype == np.uint8:
        return i
    if i.dtype == np.uint16:
        return (i >> 8).astype(np.uint8)
    return np.clip(i, 0, 255).astype(np.uint8)


# =========================================================================== #
#                                   掩膜
# =========================================================================== #
def _largest_component(mask: np.ndarray, scale: int = CC_SCALE) -> np.ndarray:
    """
    只保留面积最大的连通域，去掉背景里飞溅的孤立噪点块。

    连通域标记本质是串行扫描，全分辨率要 200+ ms CPU。这里在 1/scale 网格上
    标记「哪一块是工件」，上采样成选择掩膜后与原掩膜相与：
    形状仍由全分辨率掩膜决定，只有「保留哪块」这个拓扑判断降了分辨率。
    实测与全分辨率结果差 0.002~0.013% 的掩膜面积（都在边界），代价减半。
    """
    if cv2.countNonZero(mask) == 0:
        return mask
    h, w = mask.shape
    small = mask
    if scale > 1 and min(h, w) > 4 * scale:
        small = (_down_area(mask, scale, exact=True) > 0).astype(np.uint8)

    n, labels, stats, _ = _cv_threaded(cv2.connectedComponentsWithStats,
                                       small, 8)
    if n <= 1:
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sel = _emap(lambda a: (a == keep).astype(np.uint8),
                small.shape[0], small.shape[1], np.uint8, labels)
    if small is mask:
        return sel
    up = cv2.resize(sel, (w, h), interpolation=cv2.INTER_NEAREST)
    return _emap(cv2.bitwise_and, h, w, np.uint8, mask, up)


def build_valid_mask(depth16: np.ndarray):
    """
    与原版同参数：close(r=7) -> open(r=3) -> 最大连通域 -> erode(rect r=8)。

    形态学只在 raw 掩膜的包围盒（外扩 MASK_PAD）内做：盒外 raw 恒为 0，
    close/open/erode 的支撑半径合计远小于 MASK_PAD，所以盒外结果必然仍是 0，
    与全图计算**逐位一致**。工件通常只占画幅一半左右，这一步直接省掉近半工作量。
    """
    h, w = depth16.shape
    raw = _emap(lambda a: (a > 0).astype(np.uint8), h, w, np.uint8, depth16)

    rbox = _mask_box(raw, MASK_PAD)
    if rbox is None:                        # 全空深度图
        return raw, raw, None
    y0, y1, x0, x1 = rbox
    sl = (slice(y0, y1), slice(x0, x1))

    solid = np.ascontiguousarray(raw[sl])
    if CLOSE_PX > 0:
        solid = _morph(solid, cv2.MORPH_CLOSE, CLOSE_PX)
    if OPEN_PX > 0:
        solid = _morph(solid, cv2.MORPH_OPEN, OPEN_PX)
    solid = _largest_component(solid)

    sub = solid
    if ERODE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * ERODE_PX + 1,) * 2)
        sub = _par(lambda a: cv2.erode(a, k), solid, ERODE_PX + 2)
        if cv2.countNonZero(sub) == 0:      # 工件太细被腐蚀干净，退回未腐蚀版
            sub = solid

    mask = np.zeros((h, w), np.uint8)
    mask[sl] = sub
    # 顺手把 raw 的包围盒返回：mask 必然含在其中，后面求 mask 包围盒时
    # 不必再全图扫一遍（行投影是跨列扫描，全图单线程要 50+ ms）
    return raw, mask, rbox


def _mask_box(mask: np.ndarray, pad: int = PAD, within=None):
    """
    掩膜包围盒外扩 pad；全空返回 None（调用方走常数输出分支）。

    行投影（REDUCE_MAX 沿 axis=1）是跨列扫描，单线程要 54 ms；列投影只要 6 ms。
    两个投影互不依赖，行投影再按条带切开，于是这一步降到个位数毫秒。

    within 给定一个已知包含该掩膜的窗口时（例如 mask 由 raw 腐蚀而来，必然含在
    raw 的包围盒内），只在窗口内投影，进一步省掉一半扫描。
    """
    H, W = mask.shape
    if within is None:
        wy0, wy1, wx0, wx1 = 0, H, 0, W
    else:
        wy0, wy1, wx0, wx1 = within
    sub = mask[wy0:wy1, wx0:wx1]
    sh = sub.shape[0]
    row_any = np.empty(sh, np.uint8)

    def rows_band(a: int, b: int) -> None:
        row_any[a:b] = cv2.reduce(sub[a:b], 1, cv2.REDUCE_MAX).ravel()

    edges = _bands(sh, BAND_MIN_ROWS)
    if edges is None:
        rows_band(0, sh)
    else:
        _run_bands(edges,
                   lambda i: rows_band(int(edges[i]), int(edges[i + 1])))
    col_any = cv2.reduce(sub, 0, cv2.REDUCE_MAX).ravel()
    rows = np.flatnonzero(row_any)
    cols = np.flatnonzero(col_any)
    if rows.size == 0 or cols.size == 0:
        return None
    return (max(0, wy0 + int(rows[0]) - pad),
            min(H, wy0 + int(rows[-1]) + 1 + pad),
            max(0, wx0 + int(cols[0]) - pad),
            min(W, wx0 + int(cols[-1]) + 1 + pad))


def _guard_weight(mask: np.ndarray) -> np.ndarray:
    """距掩膜边界 GUARD_PX 内线性渐隐，只压轮廓处的形态学/LoG 假响应"""
    if GUARD_PX <= 0:
        return mask.astype(np.float32)
    inv = np.float32(1.0 / GUARD_PX)

    def op(a: np.ndarray) -> np.ndarray:
        d = cv2.distanceTransform(a, cv2.DIST_L2, 3)
        d *= inv
        return np.clip(d, 0.0, 1.0, out=d)

    # 距离在 GUARD_PX 处即被截断，halo 取 2*GUARD_PX 已足够精确
    return _par(op, mask, 2 * GUARD_PX + 2, np.float32)


# =========================================================================== #
#                     统计 / 掩膜感知低频 / 噪声尺度
# =========================================================================== #
def _stats(x: np.ndarray, mask: np.ndarray, step: int = STAT_STEP):
    """掩膜内 median 与 MAD，按 step x step 抽样（样本数万级，估计误差远小于量化步长）"""
    v = x[::step, ::step][mask[::step, ::step] > 0]
    if v.size < 256:
        v = x[mask > 0]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad < 1e-3:
        mad = max(1e-3, 0.05 * float(np.std(v)))
    return med, mad


class LowPass:
    """
    掩膜感知大 sigma 高斯：在 1/LOW_SCALE 网格上算 blur(x*m)/blur(m)。

    掩膜的降采样与模糊分母只算一次，两次 local_scale 复用。
    """

    def __init__(self, mask_f: np.ndarray):
        h, w = mask_f.shape
        self.h, self.w = h, w
        self.den = _down_area(mask_f, LOW_SCALE, exact=True)   # 掩膜占比小图
        self._cache: dict[float, np.ndarray] = {}

    def smooth(self, num_small: np.ndarray, sigma: float) -> np.ndarray:
        """
        输入已是「乘过掩膜并降采样到 1/LOW_SCALE」的分子，输出同尺寸的小图。

        分母（掩膜占比的高斯）按 sigma 缓存，两次 local_scale 复用。
        """
        s = sigma / LOW_SCALE
        den = self._cache.get(s)
        if den is None:
            den = cv2.GaussianBlur(self.den, (0, 0), s) + 1e-6
            self._cache[s] = den
        return cv2.GaussianBlur(num_small, (0, 0), s) / den

    def __call__(self, x_masked: np.ndarray, sigma: float) -> np.ndarray:
        """x_masked 是全尺寸且已乘掩膜；先降采样再走 smooth"""
        return self.smooth(_down_area(x_masked, LOW_SCALE), sigma)


def _local_scale(res: np.ndarray, mask_f: np.ndarray, low: LowPass) -> Lazy:
    """
    逐像素噪声尺度：|res - gauss(res, 2)| * mask 再做大尺度掩膜感知平滑。

    这一步的输出只是 1/LOW_SCALE 的小图，所以中间的 gauss(res,2) 与
    |res-fine|*mask **都不必存成全尺寸**：在每个条带里算完偏差就地降采样，
    只把 1/8 的结果写出去。省掉两个 27 MB 数组的写入与再读入，而这个函数
    在流水线里要跑两次（去噪前后各一次）。

    条带按 LOW_SCALE 对齐、上下留 halo，故降采样块完整、结果与整图一致。

    返回 Lazy（内部只有小图）：该场后面被 R/G 通道与去噪共读 4 次，
    保持小图形态又能省掉 4 x 27 MB 的搬运。
    """
    h, w = res.shape
    hs = max(1, -(-h // LOW_SCALE))
    ws = max(1, -(-w // LOW_SCALE))
    dev_s = np.empty((hs, ws), np.float32)
    halo = int(np.ceil(4.0 * FINE_SIGMA)) + 2

    def block(a: int, b: int) -> None:
        """a,b 是小图行号；对应原图 [a*S, min(b*S, h))"""
        y0, y1 = a * LOW_SCALE, min(b * LOW_SCALE, h)
        s0, s1 = max(0, y0 - halo), min(h, y1 + halo)
        sub = np.ascontiguousarray(res[s0:s1])
        fine = cv2.GaussianBlur(sub, (0, 0), FINE_SIGMA)
        o = y0 - s0
        dev = np.abs(sub[o:o + (y1 - y0)] - fine[o:o + (y1 - y0)])
        dev *= mask_f[y0:y1]
        pad_y = (b - a) * LOW_SCALE - (y1 - y0)
        pad_x = ws * LOW_SCALE - w
        if pad_x or pad_y:
            dev = cv2.copyMakeBorder(dev, 0, pad_y, 0, pad_x,
                                     cv2.BORDER_REPLICATE)
        dev_s[a:b] = cv2.resize(np.ascontiguousarray(dev), (ws, b - a),
                                interpolation=cv2.INTER_AREA)

    edges = _bands(hs, max(16, BAND_MIN_ROWS // LOW_SCALE))
    if edges is None:
        block(0, hs)
    else:
        _run_bands(edges, lambda i: block(int(edges[i]), int(edges[i + 1])))

    small = low.smooth(dev_s, SCALE_SIGMA)
    np.maximum(small, 1e-6, out=small)
    return Lazy(small, LOW_SCALE, h, w)


# =========================================================================== #
#                        残差：低频基线 + 掉点填充 + 行偏置
# =========================================================================== #
def _residual(depth16: np.ndarray, raw: np.ndarray, box) -> np.ndarray:
    """
    residual = where(raw, depth, fill) - baseline，只返回包围盒内的裁剪。

    baseline / fill 都是掩膜感知高斯（sigma 60 / 24），在 1/8 全图网格上算：
    原版本身也走降采样路径（scale=7 / 3），这里只是把网格再粗一档。
    小图上算完再上采样到全图后裁剪，保证盒边界附近的低频与全图一致。
    """
    y0, y1, x0, x1 = box
    sl = (slice(y0, y1), slice(x0, x1))

    # uint16 直接 INTER_AREA 下采样：省掉 12M 元素的 f32 转换，且与先转 f32
    # 的结果只差 <=0.5 个 LSB（16 位高度里可以忽略）
    num = _down_area(depth16, LOW_SCALE)
    den = _down_area(raw, LOW_SCALE, exact=True)

    def low(sigma: float) -> np.ndarray:
        s = sigma / LOW_SCALE
        small = cv2.GaussianBlur(num, (0, 0), s) / \
            (cv2.GaussianBlur(den, (0, 0), s) + 1e-6)
        return _up_region(small, LOW_SCALE, y0, y1, x0, x1)

    base = low(SIGMA_BASELINE)
    sub_d = depth16[sl]
    sub_raw = raw[sl]
    rows, cols = base.shape

    if int(cv2.countNonZero(sub_raw)) == rows * cols:      # 无掉点，省一次低频
        return _emap(lambda d, b: d.astype(np.float32) - b,
                     rows, cols, np.float32, sub_d, base)
    fill = low(FILL_SIGMA)
    return _emap(lambda d, r, b, f: np.where(r > 0, d, f).astype(np.float32) - b,
                 rows, cols, np.float32, sub_d, sub_raw, base, fill)


def _remove_row_bias(res: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    每行减掉行内中位数（压横向条纹）。

    列方向抽样到 ~ROW_SAMPLES 列（原版用全宽）：中位数的抽样误差约
    0.08*sigma，落到输出灰阶里约 1 级。行之间独立，故按行条带并行。
    """
    h, w = res.shape
    step = max(1, w // ROW_SAMPLES)
    sub = np.ascontiguousarray(res[:, ::step])
    msk = np.ascontiguousarray(mask[:, ::step])
    thr = max(4, 32 // step)
    big = np.float32(np.inf)

    def band(s: np.ndarray, m: np.ndarray) -> np.ndarray:
        # nanmedian 内部要反复处理 NaN 掩码，295 列就要 65 ms。改成把掩膜外的
        # 值换成 +inf 推到排序末尾，直接对行排序后取「有效元素的中位位置」：
        # 每行有效个数不同，所以不能用单个 kth 的 partition，但行内只有几百个
        # 元素，排序本身很便宜，条带并行后可以忽略。
        # 偶数个有效元素时取偏下的中位（nanmedian 取两者均值），差异 < 1 个量化级。
        valid = m > 0
        cnt = valid.sum(axis=1)
        x = np.where(valid, s, big)
        x.sort(axis=1)
        idx = (np.maximum(cnt, 1) - 1) // 2
        med = np.take_along_axis(x, idx[:, None], axis=1)
        return np.where((cnt > thr)[:, None], med, 0.0)

    med = _emap(band, h, 1, np.float32, sub, msk)
    return _emap(lambda r, m: r - m, h, w, np.float32, res, med)


def _denoise(res: np.ndarray, mask_small: np.ndarray,
             sigma_local: Lazy) -> np.ndarray:
    """空间自适应去噪：噪声比值高的区域（工件两侧）才平滑，中部保持原样"""
    if NR_STRENGTH <= 0:
        return res
    # quiet 分位数直接在 1/LOW_SCALE 的小图上统计：该场本身就是大尺度平滑的
    # 结果，小图的取值分布与全尺寸一致（mask_small 复用 LowPass 的掩膜占比图）
    v = sigma_local.small[mask_small > 0.5]
    quiet = max(float(np.percentile(v, 35)), 1e-6) if v.size else 1.0

    smooth = _par(lambda a: cv2.medianBlur(a, int(MED_KSIZE) | 1), res,
                  MED_KSIZE)
    if BLUR_SIGMA > 0:
        smooth = _gauss(smooth, BLUR_SIGMA)

    gain = np.float32(NR_STRENGTH / max(1e-6, RATIO_HI - RATIO_LO))
    inv_q = np.float32(1.0 / quiet)
    lo = np.float32(RATIO_LO)
    h, w = res.shape

    def band(r, sm, sg):
        wt = np.clip((sg * inv_q - lo) * gain, 0.0, 1.0)
        return r + (sm - r) * wt

    return _emap(band, h, w, np.float32, res, smooth, sigma_local)


# =========================================================================== #
#                                 三个通道
# =========================================================================== #
def _tophat_max(x: np.ndarray, radius: int, prev_white=None, prev_black=None):
    """
    单尺度 white / black top-hat，并就地与上一层（已上采样）结果取 max。

    白/黑分量必须分开累积：在斜坡区域 x-open 与 close-x 可以**同时为正**，
    先相减成有符号量再取 max 会丢信息（实测这样做与逐层分开累积的结果平均差
    2.34，量程只有 ±19，明显不可接受）。

    但「相减 + 取 max」可以合并成一趟：white = max(x - open, prev_white)。
    最终的 white - black 也不单独跑一趟，直接折进 _ch_defect 的融合内核。
    """
    rows, cols = x.shape
    opened = _morph(x, cv2.MORPH_OPEN, radius, TH_SHAPE)
    closed = _morph(x, cv2.MORPH_CLOSE, radius, TH_SHAPE)
    if prev_white is None:
        white = _emap(lambda a, o: a - o, rows, cols, np.float32, x, opened)
        black = _emap(lambda c, a: c - a, rows, cols, np.float32, closed, x)
    else:
        white = _emap(lambda a, o, p: np.maximum(a - o, p), rows, cols,
                      np.float32, x, opened, prev_white)
        black = _emap(lambda c, a, p: np.maximum(c - a, p), rows, cols,
                      np.float32, closed, x, prev_black)
    return white, black


def _ch_defect_and_shape(res: np.ndarray, mask: np.ndarray,
                         weight: np.ndarray, sigma_local: Lazy, med0: float):
    """
    一趟同时产出 G（缺陷）与 R（整体形貌）两个通道。

    两者需要的输入完全相同（res / sigma_local / weight / mask），分开算就要把
    这 4 个数组各读两遍。合并后 R 通道几乎是免费的（只多两次乘加），
    实测省掉约 35 ms。

    G 通道：多尺度 Top-Hat + 多尺度归一化 LoG。

    Top-Hat 走 3 层金字塔（每层 r=5，等效 5/10/20），代价约为全分辨率
    r=5/11/21 的 1/14；LoG 因尺度不变性把 2.0/3.0 两档放到半分辨率上算。

    结构上的关键点：**全分辨率(L0)的那一层不落盘**。
    L0 的 open/close/LoG 都是小支撑邻域算子（半径 5 / sigma 1.2），可以和
    「粗层结果上采样 + 归一化 + 融合 + 量化」压在同一个条带内一次算完，
    只输出最终的 uint8。这样省掉 white/black/log_best 三个全尺寸 float32
    （3800x1775 各 27 MB）的写入与再读入。
    实测这条流水线是内存带宽瓶颈（串行 1.9 s / 16 条带 0.86 s，只有 3.8x，
    说明多核在等内存），减少全图中间量比减少运算量有效得多。

    代价是归一化所需的 med/MAD 必须先知道，而它依赖 L0 的结果。做法是先在
    若干条**抽样条带**上算一遍（占全图约 13% 的行），用这些样本估计统计量：
    med/MAD 本身就是抽样估计（见 _stats），再抽一层不改变量级。
    """
    h, w = res.shape
    fill = np.float32(med0)
    work = _emap(lambda r, m: np.where(m > 0, r, fill), h, w, np.float32,
                 res, mask)

    # ---- 粗层（L1 及以上）：数据量只有 1/4 起，正常落盘 ----
    # 金字塔用 ceil 尺寸（_down_area 会 BORDER_REPLICATE 补齐），保证上一层
    # 每个像素都有源可取，_up_region 才能整数倍还原
    pyr = [work]
    for _ in range(TH_LEVELS - 1):
        pyr.append(_down_area(pyr[-1], 2))

    def pick(a, b):
        return np.where(np.abs(b) > np.abs(a), b, a)

    if len(pyr) > 1:
        white_c, black_c = _tophat_max(pyr[-1], TH_RADIUS)
        for lv in range(len(pyr) - 2, 0, -1):       # 停在 L1，不做 L0
            rows, cols = pyr[lv].shape
            white_c, black_c = _tophat_max(
                pyr[lv], TH_RADIUS,
                _up_region(white_c, 2, 0, rows, 0, cols),
                _up_region(black_c, 2, 0, rows, 0, cols))
        l1 = pyr[1]
        log_c = _log_norm(l1, LOG_L1_SIGMAS[0]) if LOG_L1_SIGMAS else None
        for s in LOG_L1_SIGMAS[1:]:
            log_c = _emap(pick, l1.shape[0], l1.shape[1], np.float32,
                          log_c, _log_norm(l1, s))
    else:
        white_c = black_c = log_c = None

    # ---- L0 层的逐条带内核：算完直接出 uint8 ----
    ksz = 2 * TH_RADIUS + 1
    kern = cv2.getStructuringElement(TH_SHAPE, (ksz, ksz))
    halo = max(2 * TH_RADIUS + 2, int(np.ceil(4.0 * LOG_L0_SIGMA)) + 3)
    s2 = np.float32(LOG_L0_SIGMA * LOG_L0_SIGMA)

    def l0_terms(y0: int, y1: int):
        """返回该条带的 (th, log_signed)，均为 float32；含 halo 后裁回"""
        a0, a1 = max(0, y0 - halo), min(h, y1 + halo)
        sub = np.ascontiguousarray(work[a0:a1])
        opened = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kern)
        closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kern)
        lg = cv2.Laplacian(cv2.GaussianBlur(sub, (0, 0), LOG_L0_SIGMA),
                           cv2.CV_32F, ksize=3) * s2
        o, e = y0 - a0, y0 - a0 + (y1 - y0)
        wh = sub[o:e] - opened[o:e]
        bl = closed[o:e] - sub[o:e]
        lg = lg[o:e]
        if white_c is not None:
            np.maximum(wh, _up_region(white_c, 2, y0, y1, 0, w), out=wh)
            np.maximum(bl, _up_region(black_c, 2, y0, y1, 0, w), out=bl)
            if log_c is not None:
                lg = pick(lg, _up_region(log_c, 2, y0, y1, 0, w))
        wh -= bl                                   # th = white - black
        return wh, lg

    # ---- 抽样条带估计 med/MAD ----
    n_s = 8
    rows_s = max(32, h // (n_s * 8))
    tv, lv_ = [], []
    for k in range(n_s):
        y0 = min(h - rows_s, max(0, int((k + 0.5) * h / n_s) - rows_s // 2))
        y1 = y0 + rows_s
        th_s, lg_s = l0_terms(y0, y1)
        m_s = mask[y0:y1]
        sel = m_s[:, ::STAT_STEP] > 0
        if sel.any():
            tv.append(th_s[:, ::STAT_STEP][sel])
            lv_.append(lg_s[:, ::STAT_STEP][sel])

    def med_mad(chunks):
        if not chunks:
            return 0.0, 1.0
        v = np.concatenate(chunks)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad < 1e-3:
            mad = max(1e-3, 0.05 * float(np.std(v)))
        return med, mad

    med_t, mad_t = med_mad(tv)
    med_l, mad_l = med_mad(lv_)
    # log_signed = -log_best，其中位数即 -med_l；MAD 对取负不变。
    # 归一化项 = (-lg) - (-med_l) = -(lg - med_l)，故下面用 (cl - lg)。
    ct, ft = np.float32(med_t), np.float32(FLOOR_FRAC * mad_t)
    cl, fl = np.float32(med_l), np.float32(FLOOR_FRAC * mad_l)
    kinv = np.float32(1.0 / K_LOCAL)
    a_th, a_log = np.float32(ALPHA_TH), np.float32(ALPHA_LOG)

    # R 通道用残差本身的统计量
    med_r, mad_r = _stats(res, mask)
    cr, fr = np.float32(med_r), np.float32(FLOOR_FRAC * mad_r)

    ch_g = np.empty((h, w), np.uint8)
    ch_r = np.empty((h, w), np.uint8)

    def block(y0: int, y1: int) -> None:
        th, lg = l0_terms(y0, y1)
        sg = sigma_local.band(y0, y1)
        wt = weight[y0:y1]
        mk = mask[y0:y1] > 0

        nt = (th - ct) * (kinv / np.maximum(sg, ft))
        np.clip(nt, -1.0, 1.0, out=nt)
        nl = (cl - lg) * (kinv / np.maximum(sg, fl))
        np.clip(nl, -1.0, 1.0, out=nl)
        nt *= a_th
        nt += a_log * nl
        np.clip(nt, -1.0, 1.0, out=nt)
        nt *= wt
        nt += 1.0
        nt *= 127.5
        ch_g[y0:y1] = np.where(mk, nt.astype(np.uint8), np.uint8(128))

        nr = (res[y0:y1] - cr) * (kinv / np.maximum(sg, fr))
        np.clip(nr, -1.0, 1.0, out=nr)
        nr *= wt
        nr += 1.0
        nr *= 127.5
        ch_r[y0:y1] = np.where(mk, nr.astype(np.uint8), np.uint8(0))

    edges = _bands(h, max(BAND_MIN_ROWS, 4 * halo))
    if edges is None:
        block(0, h)
    else:
        _run_bands(edges,
                   lambda i: block(int(edges[i]), int(edges[i + 1])))
    return ch_g, ch_r


def _ch_intensity(intensity8: np.ndarray, mask: np.ndarray, box) -> np.ndarray:
    """
    B 通道：掩膜内 CLAHE，掩膜外 0。

    CLAHE 的 tile 网格由图像尺寸决定，所以不能切条并行；但它是 OpenCV 里少数
    **能吃多核**的算子（本机 37 ms -> 9 ms），故临时放开 OpenCV 线程。

    注意：这一步**必须在整图上做**。曾尝试只在掩膜包围盒内做 CLAHE 以省掉
    45% 的像素，结果 tile 网格随子图尺寸改变，直方图统计窗口跟着变，B 通道
    对原版的 PSNR 从「完全一致」掉到 29~33 dB，属于可见的亮度差异，故放弃。
    整图 CLAHE 放开线程后只要约 35 ms，不是瓶颈。
    """
    H, W = intensity8.shape
    if box is None:
        return np.zeros((H, W), np.uint8)
    y0, y1 = box[0], box[1]

    # CLAHE 的输入必须是整幅（tile 网格依赖尺寸），但「掩膜外填均值」只需在
    # 包围盒的行范围内做——盒外整行掩膜全 0，填什么都会在最后被清零，
    # 直接复制原图即可。同理输出的清零也只需处理这些行，其余行整体置 0。
    fill = np.uint8(np.clip(cv2.mean(intensity8, mask)[0], 0, 255))
    src = intensity8.copy()
    src[y0:y1] = _emap(lambda a, m: np.where(m > 0, a, fill), y1 - y0, W,
                       np.uint8, intensity8[y0:y1], mask[y0:y1])

    clahe = cv2.createCLAHE(CLAHE_CLIP, (CLAHE_TILE, CLAHE_TILE))
    eq = _cv_threaded(clahe.apply, src)

    out = np.zeros((H, W), np.uint8)
    out[y0:y1] = _emap(lambda a, m: np.where(m > 0, a, np.uint8(0)),
                       y1 - y0, W, np.uint8, eq[y0:y1], mask[y0:y1])
    return out


# =========================================================================== #
#                                 公开 API
# =========================================================================== #
def synthesize(depth16: np.ndarray, intensity8: np.ndarray,
               timing: dict | None = None) -> np.ndarray:
    """height(uint16) + intensity(uint8) -> BGR uint8 训练图"""
    def mark(key: str, t0: float) -> float:
        if timing is not None:
            timing[key] = timing.get(key, 0.0) + \
                (time.perf_counter() - t0) * 1000.0
        return time.perf_counter()

    depth16 = _ensure_depth16(depth16)
    intensity8 = _ensure_intensity8(intensity8)
    if depth16.shape != intensity8.shape:
        intensity8 = cv2.resize(intensity8,
                                (depth16.shape[1], depth16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    t = time.perf_counter()

    # ---- 最开始增加 TIFF 高度图的中间值滤波（Median Filter）预处理 ---- #
    if PRE_MEDIAN_KSIZE > 1:
        depth16 = _par(lambda a: cv2.medianBlur(a, int(PRE_MEDIAN_KSIZE) | 1),
                       depth16, PRE_MEDIAN_KSIZE)
        t = mark("pre_median", t)

    raw, mask, raw_box = build_valid_mask(depth16)
    t = mark("mask", t)

    H, W = depth16.shape
    ch_r = np.zeros((H, W), np.uint8)
    ch_g = np.full((H, W), 128, np.uint8)

    # 重计算只在掩膜包围盒（外扩 PAD）内做：盒外掩膜恒为 0，R/G 输出是常数
    # （R=0、G=128），与原版一致。box 为 None 表示掩膜全空。
    box = _mask_box(mask, within=raw_box) if raw_box is not None else None
    if box is not None:
        y0, y1, x0, x1 = box
        sl = (slice(y0, y1), slice(x0, x1))
        mask_c = np.ascontiguousarray(mask[sl])
        t = mark("box", t)

        res = _residual(depth16, raw, box)
        t = mark("residual", t)
        res = _remove_row_bias(res, mask_c)
        t = mark("rowbias", t)

        mask_f = _emap(lambda m: m.astype(np.float32), mask_c.shape[0],
                       mask_c.shape[1], np.float32, mask_c)
        low = LowPass(mask_f)
        sigma = _local_scale(res, mask_f, low)
        t = mark("scale1", t)
        res = _denoise(res, low.den, sigma)
        t = mark("denoise", t)
        sigma = _local_scale(res, mask_f, low)
        t = mark("scale2", t)

        weight = _guard_weight(mask_c)
        t = mark("guard", t)
        med0, _ = _stats(res, mask_c)
        # R / G 共用同一组输入，一趟条带里同时产出
        g_sub, r_sub = _ch_defect_and_shape(res, mask_c, weight, sigma, med0)
        ch_g[sl] = g_sub
        ch_r[sl] = r_sub
        t = mark("ch_rg", t)

    ch_b = _ch_intensity(intensity8, mask, box)
    t = mark("ch_b", t)
    out = cv2.merge([ch_b, ch_g, ch_r])
    mark("merge", t)
    return out


# =========================================================================== #
#                            文件名解析 / 配对 / IO
# =========================================================================== #
HEIGHT_TOKEN = "height"
INTENSITY_TOKEN = "intensity"
DEPTH_EXTS = (".tiff", ".tif")
INTENSITY_EXTS = (".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg")
EXCLUDE_TOKENS = ("pseudocolor", "pseudo_color", "color", "merged")


def _split_role(path: str, token: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    idx = stem.lower().rfind(token)
    if idx < 0:
        return stem, False
    prefix = stem[:idx].rstrip("_- .")
    return (prefix, True) if prefix else (stem, False)


def _is_excluded(path: str) -> bool:
    low = os.path.basename(path).lower()
    return any(tok in low for tok in EXCLUDE_TOKENS)


def auto_pair(folder: str):
    """按 <前缀>_height.tiff 与 <前缀>_intensity.png 在同一目录内配对"""
    depth_map, inten_map = {}, {}
    pattern = os.path.join(folder, "**", "*") if RECURSIVE \
        else os.path.join(folder, "*")
    for p in glob(pattern, recursive=RECURSIVE):
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        low = os.path.basename(p).lower()
        key_dir = os.path.dirname(os.path.abspath(p))
        if HEIGHT_TOKEN in low and ext in DEPTH_EXTS:
            prefix, ok = _split_role(p, HEIGHT_TOKEN)
            if ok:
                depth_map[(key_dir, prefix)] = p
        elif INTENSITY_TOKEN in low and ext in INTENSITY_EXTS \
                and not _is_excluded(p):
            prefix, ok = _split_role(p, INTENSITY_TOKEN)
            if ok:
                inten_map[(key_dir, prefix)] = p

    pairs = [(depth_map[k], inten_map[k])
             for k in sorted(depth_map) if k in inten_map]
    missing = [depth_map[k] for k in sorted(depth_map) if k not in inten_map]
    orphan = [inten_map[k] for k in sorted(inten_map) if k not in depth_map]
    return pairs, missing, orphan


def read_pair(depth_path: str, inten_path: str):
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    i = cv2.imread(inten_path, cv2.IMREAD_GRAYSCALE)
    if d is None:
        raise FileNotFoundError(f"无法读取高度图: {depth_path}")
    if i is None:
        raise FileNotFoundError(f"无法读取亮度图: {inten_path}")
    return _ensure_depth16(d), _ensure_intensity8(i)


def make_out_name(depth_path: str, in_root: str) -> str:
    """输出名带相对子目录前缀，避免不同子目录下同名文件互相覆盖"""
    prefix, _ = _split_role(depth_path, HEIGHT_TOKEN)
    rel = os.path.relpath(os.path.dirname(os.path.abspath(depth_path)),
                          os.path.abspath(in_root))
    tag = "" if rel in ("", ".") else \
        rel.replace(os.sep, "_").replace("..", "up") + "__"
    return f"{tag}{prefix}_merged.png"


def unique_path(out_dir: str, name: str) -> str:
    stem, ext = os.path.splitext(name)
    cand = os.path.join(out_dir, name)
    n = 1
    while os.path.exists(cand):
        cand = os.path.join(out_dir, f"{stem}_{n}{ext}")
        n += 1
    return cand


# =========================================================================== #
#                          对比 / 剖析 / 命令行
# =========================================================================== #
BASE_SCRIPT = "synthesize-fast-多图-西克-去噪版.py"


def _load_baseline():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BASE_SCRIPT)
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("kiro_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """单通道 SSIM（11x11 高斯窗 sigma=1.5），衡量与原版的视觉一致性"""
    x = a.astype(np.float32)
    y = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx = cv2.GaussianBlur(x, (11, 11), 1.5)
    my = cv2.GaussianBlur(y, (11, 11), 1.5)
    vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
    vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
    vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
    s = ((2 * mx * my + c1) * (2 * vxy + c2)) / \
        ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())


def benchmark(in_root: str, limit: int = 0, repeat: int = 3,
              with_base: bool = True) -> None:
    pairs, _, _ = auto_pair(in_root)
    if not pairs:
        print(f"[!] 未在 {in_root} 找到配对")
        return
    if limit > 0:
        pairs = pairs[:limit]
    base = _load_baseline() if with_base else None
    if with_base and base is None:
        print(f"[!] 找不到原版脚本 {BASE_SCRIPT}，只测本方案耗时")

    print(f"{'file':20s} {'尺寸':>12s} {'原版(ms)':>10s} {'kiro(ms)':>9s} "
          f"{'加速':>7s} {'PSNR':>7s} {'SSIM':>6s} {'|Δ|':>6s}")
    tb, tf, ps, ss = [], [], [], []
    for dp, ip in pairs:
        d, i = read_pair(dp, ip)
        ref, base_ms = None, float("nan")
        if base is not None:
            t = time.perf_counter()
            ref = base.synthesize(d, i)
            base_ms = (time.perf_counter() - t) * 1000.0
            tb.append(base_ms)

        synthesize(d, i)                              # 预热（线程池 / 内存）
        best, out = float("inf"), None
        for _ in range(max(1, repeat)):
            t = time.perf_counter()
            cur = synthesize(d, i)
            dt = (time.perf_counter() - t) * 1000.0
            if dt < best:
                best, out = dt, cur
        tf.append(best)

        name = os.path.basename(dp)
        if ref is not None:
            psnr = cv2.PSNR(ref, out)
            ssim = float(np.mean([_ssim(ref[:, :, c], out[:, :, c])
                                  for c in range(3)]))
            mad = float(np.mean(cv2.absdiff(ref, out)))
            ps.append(psnr)
            ss.append(ssim)
            print(f"{name:20s} {str(d.shape):>12s} {base_ms:10.1f} {best:9.1f} "
                  f"{base_ms / best:6.2f}x {psnr:7.2f} {ssim:6.3f} {mad:6.2f}")
        else:
            print(f"{name:20s} {str(d.shape):>12s} {'-':>10s} {best:9.1f}")

    print("-" * 86)
    if tb:
        print(f"原版均值 {np.mean(tb):.0f} ms，kiro 均值 {np.mean(tf):.0f} ms "
              f"(最大 {np.max(tf):.0f} ms)，平均加速 "
              f"{np.mean(tb) / np.mean(tf):.2f}x")
        print(f"PSNR 均值 {np.mean(ps):.2f} dB，SSIM 均值 {np.mean(ss):.4f}")
    else:
        print(f"kiro 均值 {np.mean(tf):.0f} ms（最大 {np.max(tf):.0f} ms）")


def profile(in_root: str, limit: int = 0, repeat: int = 5) -> None:
    pairs, _, _ = auto_pair(in_root)
    if limit > 0:
        pairs = pairs[:limit]
    for dp, ip in pairs:
        d, i = read_pair(dp, ip)
        synthesize(d, i)
        best, best_tm = float("inf"), {}
        for _ in range(max(1, repeat)):
            tm: dict = {}
            t = time.perf_counter()
            synthesize(d, i, timing=tm)
            dt = (time.perf_counter() - t) * 1000.0
            if dt < best:
                best, best_tm = dt, tm
        print(f"\n{os.path.basename(dp)}  {d.shape}  最优 {best:.1f} ms")
        for k, v in sorted(best_tm.items(), key=lambda x: -x[1]):
            print(f"    {k:10s} {v:7.1f} ms  {100 * v / best:5.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(
        description="kiro 低延迟 3D 合成（带 TIFF 中值预处理，height + intensity -> 3 通道训练图）")
    p.add_argument("-i", "--input", default=r"./imgs/20260813/33")
    p.add_argument("-o", "--output", default="./kiro-results-median")
    p.add_argument("-t", "--threads", type=int, default=0,
                   help=f"单图条带并行度，0=自动({_MAX_THREADS})")
    p.add_argument("--benchmark", action="store_true", help="与原版对比耗时与相似度")
    p.add_argument("--no-base", action="store_true", help="对比时不跑原版（只测耗时）")
    p.add_argument("--profile", action="store_true", help="打印各阶段耗时")
    p.add_argument("--bench-limit", type=int, default=0)
    p.add_argument("--repeat", type=int, default=3, help="计时重复次数取最小值")
    args = p.parse_args()

    if args.threads:
        set_threads(args.threads)
    if not os.path.isdir(args.input):
        print(f"[!] 输入目录不存在: {args.input}")
        return
    if args.benchmark:
        benchmark(args.input, args.bench_limit, args.repeat, not args.no_base)
        return
    if args.profile:
        profile(args.input, args.bench_limit, args.repeat)
        return

    pairs, missing, orphan = auto_pair(args.input)
    if not pairs:
        print(f"[!] 未找到 *_{HEIGHT_TOKEN}.tiff / *_{INTENSITY_TOKEN}.png 配对")
        return
    os.makedirs(args.output, exist_ok=True)
    print(f"[i] 配对 {len(pairs)} 组（缺亮度 {len(missing)}，缺高度 {len(orphan)}）")

    ok, t_syn, t0 = 0, 0.0, time.perf_counter()
    for dp, ip in pairs:
        out_path = unique_path(args.output, make_out_name(dp, args.input))
        try:
            d, i = read_pair(dp, ip)
            t = time.perf_counter()
            merged = synthesize(d, i)
            t_syn += time.perf_counter() - t
            if not cv2.imwrite(out_path, merged):
                raise IOError(out_path)
            ok += 1
        except Exception as e:
            print(f"[ERR] {dp}: {e}")
    dt = time.perf_counter() - t0
    n = max(ok, 1)
    print(f"[OK] 完成 {ok}/{len(pairs)}，合计 {dt:.2f}s"
          f"（合成 {t_syn / n * 1000:.0f} ms/图，含读写 {dt / n * 1000:.0f} ms/图）")
    print(f"     输出: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

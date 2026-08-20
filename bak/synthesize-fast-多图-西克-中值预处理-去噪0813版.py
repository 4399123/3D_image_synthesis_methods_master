"""
批量版：TIFF 高度图中值预处理 + 3 通道图像合成。

针对高度图中的脉冲散点、细短水平条纹，在进入现有合成算法前执行 5x5 中值滤波。
为了避免无效背景值 0 参与排序后侵蚀工件边缘，预处理分三步：
  1. 从原始高度图提取并清理工件主轮廓；
  2. 先对无效像素做掩膜感知的局部估值，再执行中值滤波；
  3. 将主轮廓外重新置 0，保证滤波结果不会扩散到背景。

预处理后的 TIFF 只在内存中使用，不写入磁盘；最终只保存 *_merged.png。
后续合成算法、固定参数和包围盒加速均直接复用：
    synthesize-fast-多图-西克-去噪版.py

用法：
    python synthesize-fast-多图-西克-中值预处理-去噪版.py -i ./imgs -o ./results
    python synthesize-fast-多图-西克-中值预处理-去噪版.py -i ./imgs -o ./results -j 1
"""

import argparse
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


BASE_SCRIPT = "synthesize-fast-多图-西克-去噪版.py"
MEDIAN_KSIZE = 5
PREFILL_SIGMA = 2.0


def _load_base():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BASE_SCRIPT)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到基础合成脚本: {path}")
    spec = importlib.util.spec_from_file_location("sick_fast_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


def _object_support(depth16: np.ndarray) -> np.ndarray:
    """提取工件主轮廓，用于限制中值滤波结果，防止高度值扩散到背景。"""
    raw = (depth16 > 0).astype(np.uint8)
    if not raw.any():
        return raw

    if base.CLOSE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * base.CLOSE_PX + 1,) * 2)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    if base.OPEN_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * base.OPEN_PX + 1,) * 2)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    return base._largest_component(raw)


def median_preprocess_depth(depth16: np.ndarray) -> np.ndarray:
    """
    对 uint16 TIFF 高度图执行 5x5 中值滤波。

    中值滤波前先为轮廓内部的 0 值孔洞估计局部高度，避免无效值 0 参与中值排序；
    滤波后将工件主轮廓外重新置 0，保持背景和原合成算法的无效值约定不变。
    """
    depth16 = base._ensure_depth16(depth16)
    raw_mask = (depth16 > 0).astype(np.uint8)
    if not raw_mask.any():
        return depth16.copy()

    support = _object_support(depth16)
    depth_f = depth16.astype(np.float32)
    estimated = base._mask_aware_gaussian(depth_f, raw_mask, PREFILL_SIGMA)
    filled = np.where(raw_mask > 0, depth_f, estimated)
    filled16 = np.clip(np.rint(filled), 0, 65535).astype(np.uint16)

    filtered = cv2.medianBlur(filled16, MEDIAN_KSIZE)
    filtered[support == 0] = 0
    return filtered


def read_pair(depth_path: str, intensity_path: str):
    """读取图像；只对 TIFF 高度图执行预处理中值滤波。"""
    depth16, intensity8 = base.read_pair(depth_path, intensity_path)
    return median_preprocess_depth(depth16), intensity8


def synthesize(depth16: np.ndarray, intensity8: np.ndarray) -> np.ndarray:
    """先对高度图做中值预处理，再执行原快速合成算法。"""
    return base.synthesize(median_preprocess_depth(depth16), intensity8)


def process_pair(depth_path: str, intensity_path: str, out_path: str) -> str:
    depth16, intensity8 = read_pair(depth_path, intensity_path)
    merged = base.synthesize(depth16, intensity8)
    if not cv2.imwrite(out_path, merged):
        raise IOError(f"写出失败: {out_path}")
    return out_path


def _worker(task):
    depth_path, intensity_path, out_path = task
    cv2.setNumThreads(base.THREADS_PER_WORKER)
    try:
        return process_pair(depth_path, intensity_path, out_path), None
    except Exception as exc:
        return depth_path, str(exc)


def _build_tasks(pairs, in_root: str, out_dir: str):
    """在主进程预先分配输出路径，避免并行进程发生重名竞争。"""
    tasks = []
    taken = set()
    for depth_path, intensity_path in pairs:
        name = base.make_out_name(depth_path, in_root)
        out_path = base.unique_path(out_dir, name)
        while out_path in taken:
            stem, ext = os.path.splitext(out_path)
            out_path = f"{stem}_x{ext}"
        taken.add(out_path)
        tasks.append((depth_path, intensity_path, out_path))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="批量 TIFF 高度图中值预处理 + 亮度图 -> 3 通道训练图")
    parser.add_argument(
        "--input", "-i",
        default=r"D:\E\github_zl\3D_image_synthesis_methods_master\imgs\cs3",
        help="输入文件夹（含 *_height.tiff 与 *_intensity.png）")
    parser.add_argument(
        "--output", "-o",
        default="./results-fast-median-edgeclean",
        help="最终合成图输出文件夹")
    parser.add_argument(
        "--jobs", "-j", type=int, default=0,
        help=f"并行进程数，0=自动({base.WORKERS})，1=串行")
    args = parser.parse_args()

    in_root = args.input
    if not os.path.isdir(in_root):
        print(f"[!] 输入目录不存在: {in_root}")
        return

    pairs, missing, orphan = base.auto_pair(in_root)
    if not pairs:
        print(f"[!] 未在 {in_root} 找到 *_height.tiff / *_intensity.png 配对")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"[i] 配对成功 {len(pairs)} 组，中值核 {MEDIAN_KSIZE}x{MEDIAN_KSIZE}"
          f"（缺亮度图 {len(missing)}，缺高度图 {len(orphan)}）")
    for path in missing:
        print(f"    [skip] 缺 *_intensity: {path}")
    for path in orphan:
        print(f"    [skip] 缺 *_height: {path}")

    tasks = _build_tasks(pairs, in_root, args.output)
    workers = 1 if args.jobs == 1 else (args.jobs or base.WORKERS)
    workers = max(1, min(workers, len(tasks)))

    ok = 0
    errors = []
    if workers == 1:
        cv2.setNumThreads(0)
        for depth_path, intensity_path, out_path in tqdm(
                tasks, desc="median + synthesize"):
            try:
                process_pair(depth_path, intensity_path, out_path)
                ok += 1
            except Exception as exc:
                errors.append((depth_path, str(exc)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_worker, tasks)
            for result, error in tqdm(
                    results, total=len(tasks), desc=f"median + synthesize x{workers}"):
                if error is None:
                    ok += 1
                else:
                    errors.append((result, error))

    for path, error in errors:
        print(f"[ERR] {path}: {error}")
    print(f"[OK] 完成 {ok}/{len(pairs)}，输出目录: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

"""
西克 (SICK) Ranger3 多图沿轴向拼接 3D 点云生成与可视化脚本

功能：
1. 自动扫描输入目录（例如 ./imgs/20260813/32），提取所有图像组。
2. 按文件名中的数字（0, 1, 2, ..., 10）从小到大自然排序。
3. 自动匹配每张切片图对应的 _height 高度图、_intensity 强度图与 .xml 标定文件。
4. 解析 XML 标定参数，沿扫描运动方向（Y 轴 / 燃料棒长度方向）累积物理偏移量进行无缝拼合。
5. 支持多种上色模式：
   - intensity: 按相机拍摄的表面灰度反射率赋色
   - height: 按 Z 高度热力图伪彩赋色
   - frame: 按不同切片图分配不同颜色（直观查看各段拼接边界）
   - gray: 统一浅灰色
6. 提供全局体素下采样 (Voxel Downsampling) 与 .ply 文件导出功能，支持 Open3D 3D 交互显示。

用法示例：
    # 1. 默认拼接 ./imgs/20260813/32 下的所有切片并弹窗交互查看
    python 点云生成-多图拼接.py -i ./imgs/20260813/32

    # 2. 按切片分块上色查看拼接接缝
    python 点云生成-多图拼接.py -i ./imgs/20260813/32 --color-mode frame

    # 3. 按高度 Z 伪彩上色
    python 点云生成-多图拼接.py -i ./imgs/20260813/32 --color-mode height

    # 4. 拼接并导出为 ply 文件（无头模式）
    python 点云生成-多图拼接.py -i ./imgs/20260813/32 --save-ply results/fuel_rod_32_stitched.ply --no-show
"""

import os
import re
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


def parse_sick_xml(xml_path: str) -> Dict[str, float]:
    """
    解析 SICK Ranger3 相机生成的 XML 元数据文件，提取 3D 空间标定参数。
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"未找到 XML 标定文件: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    params = {}
    for p in root.iter("parameter"):
        name = p.get("name")
        val = p.text
        if name and val:
            params[name.strip().lower()] = val.strip()

    scale_x = float(params.get("a axis range scale", 0.0048))
    offset_x = float(params.get("a axis range offset", 0.0))
    scale_y = float(params.get("b axis range scale", 0.0050))
    offset_y = float(params.get("b axis range offset", 0.0))
    scale_z = float(params.get("c axis range scale", 5.355e-5))
    offset_z = float(params.get("c axis range offset", 0.0))
    missing_val = float(params.get("c axis range missing value", 0.0))
    unit = params.get("unit", "millimeter")
    device_model = params.get("device model", "Unknown SICK Camera")

    return {
        "scale_x": scale_x,
        "offset_x": offset_x,
        "scale_y": scale_y,
        "offset_y": offset_y,
        "scale_z": scale_z,
        "offset_z": offset_z,
        "missing_val": missing_val,
        "unit": unit,
        "device_model": device_model,
    }


def extract_sort_key(file_stem: str) -> Tuple[int, str]:
    """
    从文件名中提取所有数字用于自然排序。例如 'img-2' -> (2, 'img-2'), 'img-10' -> (10, 'img-10')。
    """
    numbers = re.findall(r"\d+", file_stem)
    if numbers:
        return (int(numbers[-1]), file_stem)
    return (999999, file_stem)


def discover_image_groups(input_dir: str) -> List[Dict[str, Optional[str]]]:
    """
    扫描目录并根据数字序号排序，匹配每帧的 height, intensity, xml 文件。
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"输入路径不是有效目录: {input_dir}")

    all_files = os.listdir(input_dir)

    # 提取所有唯一的 stem 基础名称
    stems = set()
    for f in all_files:
        base, ext = os.path.splitext(f)
        if ext.lower() in [".tiff", ".tif", ".png", ".xml", ".dat", ".bmp"]:
            stem = re.sub(r"(_height|_intensity|_pseudocolor)$", "", base, flags=re.IGNORECASE)
            stems.add(stem)

    sorted_stems = sorted(list(stems), key=extract_sort_key)
    if not sorted_stems:
        raise FileNotFoundError(f"在目录 [{input_dir}] 下未找到任何可用的切片图像或 XML 数据！")

    # 预先在目录下找一个可用的备用 xml（以防个别切片缺失）
    fallback_xml = None
    for f in all_files:
        if f.lower().endswith(".xml"):
            fallback_xml = os.path.join(input_dir, f)
            break

    groups = []
    for stem in sorted_stems:
        # 1. 查找高度图
        height_candidates = [
            os.path.join(input_dir, f"{stem}_height.tiff"),
            os.path.join(input_dir, f"{stem}_height.tif"),
            os.path.join(input_dir, f"{stem}_height.png"),
            os.path.join(input_dir, f"{stem}.tiff"),
            os.path.join(input_dir, f"{stem}.tif"),
        ]
        height_path = next((f for f in height_candidates if os.path.isfile(f)), None)

        if not height_path:
            # 如果没有高度图则跳过
            continue

        # 2. 查找强度图
        intensity_candidates = [
            os.path.join(input_dir, f"{stem}_intensity.png"),
            os.path.join(input_dir, f"{stem}_intensity.tiff"),
            os.path.join(input_dir, f"{stem}_intensity.tif"),
            os.path.join(input_dir, f"{stem}_intensity.bmp"),
        ]
        intensity_path = next((f for f in intensity_candidates if os.path.isfile(f)), None)

        # 3. 查找 XML
        xml_candidates = [
            os.path.join(input_dir, f"{stem}.xml"),
            os.path.join(input_dir, f"{stem}.XML"),
        ]
        xml_path = next((f for f in xml_candidates if os.path.isfile(f)), fallback_xml)

        groups.append({
            "stem": stem,
            "height_path": height_path,
            "intensity_path": intensity_path,
            "xml_path": xml_path,
        })

    return groups


def generate_stitched_point_cloud(
    groups: List[Dict[str, Optional[str]]],
    color_mode: str = "intensity",
    voxel_size_per_frame: float = 0.0,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, float]]:
    """
    遍历排序后的图片列表，沿 Y 轴累积偏移，拼接生成整根燃料棒的点云。
    """
    all_points = []
    all_colors = []

    current_y_offset = 0.0  # 沿 Y 轴累积的物理位置 (mm)

    # 预定义颜色表（用于 'frame' 模式区分各切片）
    frame_palette = [
        [0.85, 0.25, 0.20],  # 红
        [0.20, 0.60, 0.85],  # 蓝
        [0.20, 0.80, 0.35],  # 绿
        [0.95, 0.70, 0.10],  # 黄
        [0.65, 0.30, 0.85],  # 紫
        [0.10, 0.85, 0.80],  # 青
        [0.90, 0.45, 0.15],  # 橙
        [0.60, 0.80, 0.20],  # 黄绿
        [0.85, 0.30, 0.60],  # 粉
        [0.40, 0.40, 0.80],  # 靛蓝
    ]

    summary_stats = {
        "total_frames": len(groups),
        "total_raw_points": 0,
        "total_y_length": 0.0,
    }

    print(f"\n开始逐帧处理并拼接 {len(groups)} 个图像切片...")

    for idx, g in enumerate(tqdm(groups, desc="拼接进度")):
        height_path = g["height_path"]
        intensity_path = g["intensity_path"]
        xml_path = g["xml_path"]

        if not xml_path:
            raise FileNotFoundError(f"帧 [{g['stem']}] 无法获取到有效的 XML 标定文件！")

        calib = parse_sick_xml(xml_path)
        height_img = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        if height_img is None:
            continue

        intensity_img = None
        if intensity_path and os.path.exists(intensity_path):
            intensity_img = cv2.imread(intensity_path, cv2.IMREAD_GRAYSCALE)

        H, W = height_img.shape
        frame_y_length = H * calib["scale_y"]  # 当前帧覆盖的物理长度 (mm)

        # 过滤无效背景点
        missing_val = calib["missing_val"]
        valid_mask = (height_img > missing_val)
        valid_count = np.count_nonzero(valid_mask)
        if valid_count == 0:
            current_y_offset += frame_y_length
            continue

        y_idx, x_idx = np.nonzero(valid_mask)
        z_raw = height_img[valid_mask]

        # 计算 3D 物理空间坐标 (单位: mm)
        # X: 截面宽度方向
        # Y: 运动方向 (加上此前切片累积的 global Y 偏移)
        # Z: 径向深度/高程方向
        X = calib["offset_x"] + x_idx.astype(np.float32) * calib["scale_x"]
        Y = (current_y_offset + calib["offset_y"]) + y_idx.astype(np.float32) * calib["scale_y"]
        Z = calib["offset_z"] + z_raw.astype(np.float32) * calib["scale_z"]

        pts_frame = np.stack([X, Y, Z], axis=-1)
        summary_stats["total_raw_points"] += len(pts_frame)

        # 赋色
        if color_mode == "intensity" and intensity_img is not None:
            gray = intensity_img[valid_mask].astype(np.float32) / 255.0
            cols_frame = np.stack([gray, gray, gray], axis=-1)
        elif color_mode == "frame":
            palette_color = frame_palette[idx % len(frame_palette)]
            cols_frame = np.tile(palette_color, (len(pts_frame), 1))
        elif color_mode in ("height", "pseudo"):
            cols_frame = Z.reshape(-1, 1)
        else:
            cols_frame = np.full((len(pts_frame), 3), 0.75, dtype=np.float32)

        # 如果开启了单帧局部降采样
        if voxel_size_per_frame > 0:
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(pts_frame)
            if color_mode in ("height", "pseudo"):
                pcd_temp.colors = o3d.utility.Vector3dVector(np.hstack([cols_frame, cols_frame, cols_frame]))
            else:
                pcd_temp.colors = o3d.utility.Vector3dVector(cols_frame)
            pcd_temp = pcd_temp.voxel_down_sample(voxel_size=voxel_size_per_frame)
            pts_frame = np.asarray(pcd_temp.points, dtype=np.float32)
            cols_frame = np.asarray(pcd_temp.colors, dtype=np.float32)
            if color_mode in ("height", "pseudo"):
                cols_frame = cols_frame[:, 0:1]

        all_points.append(pts_frame)
        all_colors.append(cols_frame)

        # 累积当前帧的 Y 轴长度
        current_y_offset += frame_y_length

    summary_stats["total_y_length"] = current_y_offset

    if not all_points:
        raise ValueError("未能提取到任何有效的点云数据！")

    # 拼合所有点云
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    # 全局高程伪彩处理
    if color_mode in ("height", "pseudo"):
        Z_all = merged_points[:, 2]
        z_min, z_max = Z_all.min(), Z_all.max()
        z_norm = np.clip((Z_all - z_min) / (z_max - z_min + 1e-8), 0.0, 1.0)
        z_uint8 = (z_norm * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(z_uint8, cv2.COLORMAP_JET)
        merged_colors = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB).squeeze() / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    return pcd, summary_stats


def main():
    parser = argparse.ArgumentParser(description="西克 (SICK) 3D 相机多切片图像沿轴向无缝拼接点云生成工具")
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default="./imgs/20260813/32",
        help="包含连续扫描切片图像与 XML 的输入文件夹路径 (默认: ./imgs/20260813/31)",
    )
    parser.add_argument(
        "--color-mode",
        choices=["intensity", "height", "frame", "gray"],
        default="intensity",
        help="点云着色模式：intensity (真实反射强度灰度), height (高度伪彩热力图), frame (按每张子图分配不同颜色标记边界), gray (纯灰)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="体素滤波下采样大小 (单位: mm)，用于加速显示与减小内存占用。默认 0.05mm (设为 0 则不进行下采样)",
    )
    parser.add_argument(
        "--save-ply",
        type=str,
        default=None,
        help="可选：拼接后点云保存为 .ply 文件的路径 (例如 ./results/fuel_rod_32.ply)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="是否跳过 3D 弹窗交互显示（用于后台批处理或无图形界面环境）",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("西克 SICK 多图轴向拼接 3D 点云生成程序启动")
    print(f"扫描目录: {args.input_dir}")
    print(f"着色模式: {args.color_mode} | 体素下采样: {args.voxel_size} mm")
    print("=" * 65)

    # 1. 扫描与按序号自然排序
    groups = discover_image_groups(args.input_dir)
    print(f"成功检索并匹配到 {len(groups)} 组连续切片数据:")
    for idx, g in enumerate(groups):
        print(f"  [{idx+1:02d}] 前缀: {g['stem']:<10} | 高度: {os.path.basename(g['height_path'])} | XML: {os.path.basename(g['xml_path']) if g['xml_path'] else '无'}")

    # 2. 逐帧生成点云并沿 Y 轴拼接
    pcd, stats = generate_stitched_point_cloud(
        groups=groups,
        color_mode=args.color_mode,
        voxel_size_per_frame=0.0,
    )

    # 3. 统计与信息打印
    bbox = pcd.get_axis_aligned_bounding_box()
    min_pt = bbox.get_min_bound()
    max_pt = bbox.get_max_bound()
    dims = max_pt - min_pt

    print("\n" + "=" * 65)
    print("【拼接完成统计信息】")
    print(f"  - 参与拼接切片总数 : {stats['total_frames']} 帧")
    print(f"  - 原始有效点云总数 : {stats['total_raw_points']:,} 点")
    print(f"  - 燃料棒总物理长度 (Y): {stats['total_y_length']:.2f} mm (跨度 {min_pt[1]:.2f} ~ {max_pt[1]:.2f} mm)")
    print(f"  - 燃料棒横向宽度   (X): {dims[0]:.2f} mm (跨度 {min_pt[0]:.2f} ~ {max_pt[0]:.2f} mm)")
    print(f"  - 径向高度范围     (Z): {dims[2]:.2f} mm (跨度 {min_pt[2]:.2f} ~ {max_pt[2]:.2f} mm)")
    print("=" * 65)

    # 4. 全局体素下采样 (加速 3D 可视化交互)
    if args.voxel_size > 0:
        print(f"\n正在执行体素网格滤波下采样 (voxel_size={args.voxel_size} mm)...", end="", flush=True)
        pcd_display = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f" 完成！点数由 {len(pcd.points):,} 精简至 {len(pcd_display.points):,} 点")
    else:
        pcd_display = pcd

    # 5. 保存点云到 PLY 文件
    if args.save_ply:
        out_dir = os.path.dirname(args.save_ply)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        o3d.io.write_point_cloud(args.save_ply, pcd)
        print(f"\n完整燃料棒点云已成功保存至: {args.save_ply}")

    # 6. 3D 可视化交互
    if not args.no_show:
        print("\n正在打开 Open3D 可视化窗口 (按 Q 或 Esc 退出, 鼠标左键旋转, 滚轮缩放)...\n")
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=min_pt)
        o3d.visualization.draw_geometries(
            [pcd_display, axis],
            window_name=f"Fuel Rod 3D Stitched Point Cloud - {os.path.basename(os.path.normpath(args.input_dir))}",
            width=1400,
            height=800,
            point_show_normal=False,
        )


if __name__ == "__main__":
    main()


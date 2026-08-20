"""
西克 (SICK) Ranger3 单图 3D 点云生成与可视化脚本

功能：
1. 自动搜索与输入图片相匹配的 .xml 标定文件、_height 高度图和 _intensity 强度图。
2. 自动解析 XML 中的 a/b/c 轴比例 (scale)、偏移 (offset) 及无效值 (missing value) 等标定参数。
3. 将 16位高度图转换为精确的物理空间 3D 点云 (单位: mm)，并结合反射强度/高度伪彩赋色。
4. 使用 Open3D 进行 3D 交互显示，支持体素下采样与导出 PLY 文件。

用法示例：
    python 点云生成-单图.py -i imgs/0813CS/32-130BAOGUAN-15/img-0_height.tiff
    python 点云生成-单图.py -i imgs/0813CS/32-130BAOGUAN-15/img-0.xml --color-mode height
    python 点云生成-单图.py -i imgs/0813CS/32-130BAOGUAN-15/img-0_height.tiff --save-ply output.ply
"""

import os
import re
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


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

    # 提取 X / Y / Z (a / b / c axis) 的 scale 与 offset
    scale_x = float(params.get("a axis range scale", 0.0048))
    offset_x = float(params.get("a axis range offset", 0.0))
    scale_y = float(params.get("b axis range scale", 0.0050))
    offset_y = float(params.get("b axis range offset", 0.0))
    scale_z = float(params.get("c axis range scale", 5.355e-5))
    offset_z = float(params.get("c axis range offset", 0.0))
    missing_val = float(params.get("c axis range missing value", 0.0))
    unit = params.get("unit", "millimeter")

    device_model = params.get("device model", "Unknown SICK Camera")

    calib_data = {
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
    return calib_data


def find_matching_files(input_path: str) -> Tuple[str, Optional[str], str]:
    """
    根据输入的任意相关文件路径（如 height.tiff, intensity.png, .xml, 或是无后缀前缀），
    自动搜索并匹配出对应的 (height_path, intensity_path, xml_path)。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    dir_name = os.path.dirname(input_path) or "."
    base_name = os.path.basename(input_path)

    # 去除已知后缀得到基础 stem (例如: "img-0_height.tiff" -> "img-0")
    stem = re.sub(r"(_height|_intensity|_pseudocolor)?\.(tiff|tif|png|xml|dat|bmp)$", "", base_name, flags=re.IGNORECASE)

    # 1. 查找 XML 文件
    xml_candidates = [
        os.path.join(dir_name, f"{stem}.xml"),
        os.path.join(dir_name, f"{stem}.XML"),
    ]
    xml_path = next((f for f in xml_candidates if os.path.isfile(f)), None)
    if not xml_path:
        raise FileNotFoundError(f"在目录 [{dir_name}] 中未找到与 [{stem}] 对应的 XML 标定文件！")

    # 2. 查找 Height 高度图文件
    height_candidates = [
        os.path.join(dir_name, f"{stem}_height.tiff"),
        os.path.join(dir_name, f"{stem}_height.tif"),
        os.path.join(dir_name, f"{stem}_height.png"),
        os.path.join(dir_name, f"{stem}.tiff"),
        os.path.join(dir_name, f"{stem}.tif"),
    ]
    height_path = next((f for f in height_candidates if os.path.isfile(f)), None)
    if not height_path:
        raise FileNotFoundError(f"在目录 [{dir_name}] 中未找到与 [{stem}] 对应的高度图文件！")

    # 3. 查找 Intensity 强度图文件 (可选)
    intensity_candidates = [
        os.path.join(dir_name, f"{stem}_intensity.png"),
        os.path.join(dir_name, f"{stem}_intensity.tiff"),
        os.path.join(dir_name, f"{stem}_intensity.tif"),
        os.path.join(dir_name, f"{stem}_intensity.bmp"),
    ]
    intensity_path = next((f for f in intensity_candidates if os.path.isfile(f)), None)

    return height_path, intensity_path, xml_path


def generate_point_cloud(
    height_path: str,
    intensity_path: Optional[str],
    calib: Dict[str, float],
    color_mode: str = "intensity",
) -> o3d.geometry.PointCloud:
    """
    根据高度图、强度图以及标定参数，生成 Open3D 点云对象。
    """
    print(f"正在读取高度图: {height_path}")
    height_img = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
    if height_img is None:
        raise ValueError(f"无法读取高度图像文件: {height_path}")

    # 读取强度图
    intensity_img = None
    if intensity_path and os.path.exists(intensity_path):
        print(f"正在读取强度图: {intensity_path}")
        intensity_img = cv2.imread(intensity_path, cv2.IMREAD_GRAYSCALE)

    # 过滤无效背景点（通常 0 为 missing value）
    missing_val = calib["missing_val"]
    valid_mask = (height_img > missing_val)
    valid_count = np.count_nonzero(valid_mask)
    if valid_count == 0:
        raise ValueError("图像中没有有效的高程数据（全为背景/无效值 0）！")

    y_idx, x_idx = np.nonzero(valid_mask)
    z_raw = height_img[valid_mask]

    # 根据标定公式计算真实 3D 物理空间坐标 (单位: mm)
    # X = offset_x + col * scale_x
    # Y = offset_y + row * scale_y
    # Z = offset_z + raw_val * scale_z
    X = calib["offset_x"] + x_idx.astype(np.float64) * calib["scale_x"]
    Y = calib["offset_y"] + y_idx.astype(np.float64) * calib["scale_y"]
    Z = calib["offset_z"] + z_raw.astype(np.float64) * calib["scale_z"]

    points = np.stack([X, Y, Z], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 设置点云颜色
    if color_mode == "intensity" and intensity_img is not None:
        gray = intensity_img[valid_mask].astype(np.float64) / 255.0
        colors = np.stack([gray, gray, gray], axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif color_mode in ("height", "pseudo"):
        # 按 Z 高度归一化进行伪彩上色
        z_min, z_max = Z.min(), Z.max()
        z_norm = np.clip((Z - z_min) / (z_max - z_min + 1e-8), 0.0, 1.0)
        # 生成 Jet 伪彩
        z_uint8 = (z_norm * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(z_uint8, cv2.COLORMAP_JET)
        color_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB).squeeze() / 255.0
        pcd.colors = o3d.utility.Vector3dVector(color_rgb)
    else:
        # 默认中性浅灰色
        colors = np.full_like(points, 0.7)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def main():
    parser = argparse.ArgumentParser(description="西克 (SICK) 3D 相机单图点云生成与可视化工具")
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="imgs/20260813/32/img-6_height.tiff",
        help="输入文件路径（可为 _height.tiff, _intensity.png, .xml 或文件前缀）",
    )
    parser.add_argument(
        "--color-mode",
        choices=["intensity", "height", "gray"],
        default="intensity",
        help="点云着色模式：intensity (反射率灰度), height (高度伪彩), gray (纯灰)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="体素下采样大小 (单位: mm)，设为 0 则不进行下采样。默认 0.02mm",
    )
    parser.add_argument(
        "--save-ply",
        type=str,
        default=None,
        help="可选：将点云保存为 .ply 文件路径 (例如 output.ply)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="是否跳过弹窗显示（用于脚本批处理）",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("西克 SICK 3D 点云生成程序启动")
    print(f"目标输入: {args.input}")
    print("=" * 60)

    # 1. 自动搜索匹配的文件
    height_path, intensity_path, xml_path = find_matching_files(args.input)
    print(f"[匹配结果]")
    print(f"  - 高度图 (Height)   : {height_path}")
    print(f"  - 强度图 (Intensity): {intensity_path if intensity_path else '未找到(跳过)'}")
    print(f"  - 标定文件 (XML)    : {xml_path}")

    # 2. 解析 XML 参数
    calib = parse_sick_xml(xml_path)
    print(f"\n[标定参数解析成功 - {calib['device_model']}]")
    print(f"  - X 轴: scale = {calib['scale_x']:.8f} mm/pix, offset = {calib['offset_x']:.4f} mm")
    print(f"  - Y 轴: scale = {calib['scale_y']:.8f} mm/line, offset = {calib['offset_y']:.4f} mm")
    print(f"  - Z 轴: scale = {calib['scale_z']:.8e} mm/DN, offset = {calib['offset_z']:.4f} mm")
    print(f"  - 单位: {calib['unit']}, 无效值: {calib['missing_val']}")

    # 3. 生成点云
    pcd = generate_point_cloud(
        height_path=height_path,
        intensity_path=intensity_path,
        calib=calib,
        color_mode=args.color_mode,
    )
    total_pts = len(pcd.points)
    print(f"\n原始有效点云数量: {total_pts:,} 点")

    # 4. 可选下采样
    if args.voxel_size > 0:
        pcd_display = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"体素下采样 (voxel_size={args.voxel_size} mm) 后点数: {len(pcd_display.points):,} 点")
    else:
        pcd_display = pcd

    # 5. 保存点云 (如果指定)
    if args.save_ply:
        out_dir = os.path.dirname(args.save_ply)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        o3d.io.write_point_cloud(args.save_ply, pcd)
        print(f"点云已成功保存至: {args.save_ply}")

    # 6. 3D 可视化
    if not args.no_show:
        print("\n正在打开 3D 可视化窗口 (按 Q 或 Esc 退出, 鼠标左键旋转, 滚轮缩放)...")
        # 添加坐标轴辅助参考 (长度 5mm)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[calib['offset_x'], calib['offset_y'], calib['offset_z']])
        o3d.visualization.draw_geometries(
            [pcd_display, axis],
            window_name=f"SICK 3D Point Cloud - {os.path.basename(height_path)}",
            width=1280,
            height=720,
            point_show_normal=False,
        )


if __name__ == "__main__":
    main()

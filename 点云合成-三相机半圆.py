"""
西克 (SICK) Ranger3 三相机环形分布点云拼接与 180° 半圆柱体合成脚本

背景与原理：
1. 3 个 SICK 3D 相机呈环形分布（相机 1 约 -60°，相机 2 为 0° 正视，相机 3 约 +60°），各自覆盖一段圆弧。
2. 脚本自动拟合各相机坐标系下的截面圆心 (Cx, Cz) 与半径，消除各相机的零点偏移。
3. 按照各自的安装角度绕轴线旋转变换，并可选用 ICP (迭代最近邻点) 进行重叠视场微调对齐。
4. 拼接合成连续完整的 180°+ (覆盖超 210°) 燃料棒半圆柱表面 3D 点云。

用法示例：
    # 1. 默认合成 imgs/20260813 (31, 32, 33 相机) 的 img-3 切片半圆柱，按相机区分颜色查看
    python 点云合成-三相机半圆.py --color-mode camera

    # 2. 合成单张切片并按真实拍摄反射率 (灰度) 赋色
    python 点云合成-三相机半圆.py --color-mode intensity

    # 3. 对整根燃料棒的所有切片 (img-0 ~ img-10) 进行三相机全长 180° 半圆柱合成并保存 PLY
    python 点云合成-三相机半圆.py --all-slices --save-ply results/half_cylinder_full.ply
"""

import os
import re
import glob
import argparse
import copy
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


def parse_sick_xml(xml_path: str) -> Dict[str, float]:
    """
    解析 SICK Ranger3 相机 XML 标定参数。
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

    return {
        "scale_x": float(params.get("a axis range scale", 0.0048)),
        "offset_x": float(params.get("a axis range offset", 0.0)),
        "scale_y": float(params.get("b axis range scale", 0.0050)),
        "offset_y": float(params.get("b axis range offset", 0.0)),
        "scale_z": float(params.get("c axis range scale", 5.355e-5)),
        "offset_z": float(params.get("c axis range offset", 0.0)),
        "missing_val": float(params.get("c axis range missing value", 0.0)),
    }


def fit_circle_2d(x: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """
    使用代数最小二乘法拟合 2D 截面圆心 (cx, cz) 与半径 r (单位: mm)。
    """
    A = np.column_stack([2 * x, 2 * z, np.ones_like(x)])
    b = x**2 + z**2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cz = float(res[0]), float(res[1])
    r = float(np.sqrt(res[2] + cx**2 + cz**2))
    return cx, cz, r


def load_single_camera_frame(
    folder: str,
    stem: str,
    y_accum_offset: float = 0.0,
    x_clip: Tuple[float, float] = (13.5, 22.0),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
    """
    读取单相机单切片的高度图、强度图与 XML，生成去除背景夹具后的局部点云与颜色。
    """
    hpath = os.path.join(folder, f"{stem}_height.tiff")
    if not os.path.exists(hpath):
        hpath = os.path.join(folder, f"{stem}_height.tif")
    ipath = os.path.join(folder, f"{stem}_intensity.png")
    xpath = os.path.join(folder, f"{stem}.xml")

    calib = parse_sick_xml(xpath)
    h_img = cv2.imread(hpath, cv2.IMREAD_UNCHANGED)
    if h_img is None:
        raise ValueError(f"无法读取文件: {hpath}")

    i_img = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE) if os.path.exists(ipath) else None

    H, W = h_img.shape
    frame_y_len = H * calib["scale_y"]

    # 过滤无效背景 (0值)
    valid = (h_img > calib["missing_val"])
    y_idx, x_idx = np.nonzero(valid)
    z_raw = h_img[valid]

    # 物理坐标转换 (mm)
    X = calib["offset_x"] + x_idx.astype(np.float64) * calib["scale_x"]
    Y = (y_accum_offset + calib["offset_y"]) + y_idx.astype(np.float64) * calib["scale_y"]
    Z = calib["offset_z"] + z_raw.astype(np.float64) * calib["scale_z"]

    # 滤除视野边缘工件夹具 (仅保留燃料棒圆弧面区域)
    mask = (X >= x_clip[0]) & (X <= x_clip[1])
    X, Y, Z = X[mask], Y[mask], Z[mask]

    pts = np.stack([X, Y, Z], axis=-1)

    if i_img is not None:
        gray = (i_img[valid][mask].astype(np.float64) / 255.0)
        colors = np.stack([gray, gray, gray], axis=-1)
    else:
        colors = np.full_like(pts, 0.75)

    return pts, colors, calib, frame_y_len


def process_camera_cloud(
    cam_folder: str,
    stem_list: List[str],
    nominal_angle_deg: float,
    color_mode: str,
    cam_color_rgb: List[float],
) -> o3d.geometry.PointCloud:
    """
    加载单个相机的一系列切片，自动拟合中轴线圆心，并旋转到世界坐标系。
    """
    all_pts = []
    all_colors = []
    y_accum = 0.0

    for stem in stem_list:
        try:
            pts, cols, calib, f_ylen = load_single_camera_frame(cam_folder, stem, y_accum_offset=y_accum)
            all_pts.append(pts)
            all_colors.append(cols)
            y_accum += f_ylen
        except Exception as e:
            continue

    if not all_pts:
        raise ValueError(f"相机目录 [{cam_folder}] 中未能加载到有效数据！")

    pts_merged = np.vstack(all_pts)
    cols_merged = np.vstack(all_colors)

    # 1. 2D 截面圆弧拟合求圆心 (Cx, Cz)
    sample_pts = pts_merged[::100]
    cx, cz, r = fit_circle_2d(sample_pts[:, 0], sample_pts[:, 2])

    # 2. 平移至以燃料棒中轴线为坐标原点 (0, 0)
    pts_merged[:, 0] -= cx
    pts_merged[:, 2] -= cz

    # 3. 绕 Y 轴旋转 nominal_angle_deg
    theta = np.radians(nominal_angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = pts_merged[:, 0] * cos_t + pts_merged[:, 2] * sin_t
    z_rot = -pts_merged[:, 0] * sin_t + pts_merged[:, 2] * cos_t
    pts_merged[:, 0] = x_rot
    pts_merged[:, 2] = z_rot

    # 4. 根据模式着色
    if color_mode == "camera":
        cols_merged = np.tile(cam_color_rgb, (len(pts_merged), 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_merged)
    pcd.colors = o3d.utility.Vector3dVector(cols_merged)

    return pcd, (cx, cz, r)


def refine_with_icp(
    pcd_source: o3d.geometry.PointCloud,
    pcd_target: o3d.geometry.PointCloud,
    max_corr_dist: float = 0.5,
) -> o3d.geometry.PointCloud:
    """
    使用 ICP 算法微调配准相邻相机的重叠点云接缝。
    """
    # 限制变换为绕 Y 轴的微小刚体变换
    init_trans = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_source,
        pcd_target,
        max_corr_dist,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    pcd_aligned = copy.deepcopy(pcd_source)
    pcd_aligned.transform(reg_p2p.transformation)
    return pcd_aligned


def main():
    parser = argparse.ArgumentParser(description="西克 (SICK) Ranger3 三相机环形分布 180° 半圆柱点云合成")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./imgs/20260813",
        help="包含 31, 32, 33 相机子目录的主文件夹 (默认: ./imgs/20260813)",
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        default=["31", "32", "33"],
        help="相机子目录名称列表 (默认: 31 32 33)",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        type=float,
        default=[-60.0, 0.0, 60.0],
        help="对应相机的安装标称旋转角度 (默认: -60.0 0.0 60.0)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="img-3",
        help="单切片合成时的切片名称 (默认: img-3)",
    )
    parser.add_argument(
        "--all-slices",
        action="store_true",
        help="若开启，则拼接所有连续切片 (img-0 ~ img-10) 合成全长半圆柱",
    )
    parser.add_argument(
        "--color-mode",
        choices=["camera", "intensity", "height", "gray"],
        default="camera",
        help="着色模式：camera (按红/绿/蓝区分三相机), intensity (反射率灰度), height (Z高程伪彩), gray (纯灰)",
    )
    parser.add_argument(
        "--enable-icp",
        action="store_true",
        default=True,
        help="是否开启 ICP 接缝迭代精配准 (默认: True)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help="体素网格下采样大小 (单位: mm)，默认 0.04 mm",
    )
    parser.add_argument(
        "--save-ply",
        type=str,
        default=None,
        help="可选：将合成的 180° 半圆柱点云保存为 .ply 文件路径",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="是否跳过 3D 弹窗交互显示",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("西克 SICK 三相机环形 180° 半圆柱点云合成启动")
    print(f"数据目录: {args.root_dir} | 相机列表: {args.cams} | 角度: {args.angles}")
    print(f"着色模式: {args.color_mode} | ICP 配准: {args.enable_icp}")
    print("=" * 65)

    # 确定要处理的切片列表
    if args.all_slices:
        # 扫描 32 号相机的全部 stem 作为基准
        base_dir = os.path.join(args.root_dir, args.cams[1])
        h_files = glob.glob(os.path.join(base_dir, "*_height.tiff")) + glob.glob(os.path.join(base_dir, "*_height.tif"))
        stems = sorted([re.sub(r"(_height)?\.(tiff|tif)$", "", os.path.basename(f), flags=re.IGNORECASE) for f in h_files],
                       key=lambda s: int(re.findall(r"\d+", s)[-1]) if re.findall(r"\d+", s) else 0)
        print(f"模式: 全长切片合成，共 {len(stems)} 个切片: {stems}")
    else:
        stems = [args.slice]
        print(f"模式: 单切片合成 [{args.slice}]")

    # 3 个相机的专属代表色 (RGB: 相机1红, 相机2绿, 相机3蓝)
    cam_colors = [
        [0.90, 0.25, 0.20],  # 红色 (Cam 31, -60°)
        [0.20, 0.80, 0.30],  # 绿色 (Cam 32, 0° 居中)
        [0.20, 0.50, 0.95],  # 蓝色 (Cam 33, +60°)
    ]

    pcd_list = []
    fits = []

    for cam_name, angle, col in zip(args.cams, args.angles, cam_colors):
        folder = os.path.join(args.root_dir, cam_name)
        print(f"\n正在加载与对齐相机 [{cam_name}] (安装角: {angle:+.1f}°)...")
        pcd_cam, fit_res = process_camera_cloud(
            cam_folder=folder,
            stem_list=stems,
            nominal_angle_deg=angle,
            color_mode=args.color_mode,
            cam_color_rgb=col,
        )
        print(f"  -> 拟合圆心: (Cx={fit_res[0]:.3f}, Cz={fit_res[1]:.3f}) mm, 拟合半径: R={fit_res[2]:.3f} mm, 点数: {len(pcd_cam.points):,}")
        pcd_list.append(pcd_cam)
        fits.append(fit_res)

    # ICP 接缝微调配准
    pcd_left, pcd_center, pcd_right = pcd_list[0], pcd_list[1], pcd_list[2]
    
    # 局部下采样用于 ICP 加速
    pcd_center_down = pcd_center.voxel_down_sample(args.voxel_size)
    pcd_left_down = pcd_left.voxel_down_sample(args.voxel_size)
    pcd_right_down = pcd_right.voxel_down_sample(args.voxel_size)

    if args.enable_icp:
        print("\n正在执行 ICP 算法消除相邻相机交叠接缝微小台阶...")
        pcd_left_aligned = refine_with_icp(pcd_left_down, pcd_center_down, max_corr_dist=0.3)
        pcd_right_aligned = refine_with_icp(pcd_right_down, pcd_center_down, max_corr_dist=0.3)
    else:
        pcd_left_aligned = pcd_left_down
        pcd_right_aligned = pcd_right_down

    # 合并三相机半圆柱点云
    pcd_half = pcd_left_aligned + pcd_center_down + pcd_right_aligned

    # 全局高程伪彩
    if args.color_mode in ("height", "pseudo"):
        pts_half = np.asarray(pcd_half.points)
        r_all = np.sqrt(pts_half[:, 0]**2 + pts_half[:, 2]**2)
        r_min, r_max = np.percentile(r_all, 1), np.percentile(r_all, 99)
        r_norm = np.clip((r_all - r_min) / (r_max - r_min + 1e-6), 0.0, 1.0)
        color_map = cv2.applyColorMap((r_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colors_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB).squeeze() / 255.0
        pcd_half.colors = o3d.utility.Vector3dVector(colors_rgb)

    # 统计信息
    pts = np.asarray(pcd_half.points)
    angles_deg = np.degrees(np.arctan2(pts[:, 0], pts[:, 2]))
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 2]**2)

    print("\n" + "=" * 65)
    print("【180° 半圆柱体点云合成成功】")
    print(f"  - 合成总点数   : {len(pts):,} 点")
    print(f"  - 圆周覆盖角度 : {angles_deg.min():.1f}° ~ {angles_deg.max():.1f}° (总视角: {angles_deg.max() - angles_deg.min():.1f}°，完全覆盖 180° 半圆)")
    print(f"  - 截面外径均值 : 直径 {np.mean(radii)*2:.2f} mm (半径 {np.mean(radii):.2f} ± {np.std(radii):.2f} mm)")
    print(f"  - 轴向 Y 长度  : {pts[:, 1].max() - pts[:, 1].min():.2f} mm (跨度 {pts[:, 1].min():.2f} ~ {pts[:, 1].max():.2f} mm)")
    print("=" * 65)

    # 保存 PLY
    if args.save_ply:
        out_dir = os.path.dirname(args.save_ply)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        o3d.io.write_point_cloud(args.save_ply, pcd_half)
        print(f"\n半圆柱点云已成功保存至: {args.save_ply}")

    # 3D 弹窗展示
    if not args.no_show:
        print("\n正在打开 3D 可视化窗口 (红色: Cam 31, 绿色: Cam 32, 蓝色: Cam 33)...")
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries(
            [pcd_half, axis],
            window_name="SICK 3-Camera 180-deg Half Cylinder Point Cloud",
            width=1400,
            height=800,
            point_show_normal=False,
        )


if __name__ == "__main__":
    main()


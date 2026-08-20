import cv2
import numpy as np

# 读取高度TIFF，保留原始16位uint16
# img_path = "imgs\img-0_height.tiff"
img_path = r"imgs\20260813/33/img-2_height.tiff"
depth_tif = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
print(depth_tif.dtype)  # uint16
h, w = depth_tif.shape

# 相机标定参数（从LJ软件复制）
ScaleZ = 0.005    # mm/计数
PitchX = 0.5     # X单像素宽度mm
PitchY = 0.5    # Y扫描步距mm

# 转换为真实高度mm
z_real = depth_tif.astype(np.float32) * ScaleZ

# 生成完整点云 N×3 (X,Y,Z)
u, v = np.meshgrid(np.arange(w), np.arange(h))
X = u * PitchX
Y = v * PitchY
point_cloud = np.stack([X, Y, z_real], axis=-1).reshape(-1, 3)

# 过滤无效点（原始值=0为遮挡）
mask = depth_tif > 0
valid_cloud = point_cloud[mask.reshape(-1)]

# 保存PCD点云
np.savetxt(img_path.replace(".tiff",".txt"), valid_cloud)
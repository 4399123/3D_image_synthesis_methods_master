import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

output_dir=r'./results/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

depth_path=r'./imgs/1.tiff'
png_path=r'./imgs/1.png'

# 读取
depth16 = cv2.imread(depth_path, -1)
intensity = cv2.imread(png_path, 0)

name = os.path.basename(depth_path).split('.')[0]

# ===== 1. 深度归一化（简化版）=====
valid = depth16[depth16 > 0]

min_d = np.percentile(valid, 1)
max_d = np.percentile(valid, 99)

depth = np.clip((depth16 - min_d) / (max_d - min_d), 0, 1)
depth = (depth * 255).astype(np.uint8)

# ===== 2. 局部对比（核心）=====
blur = cv2.GaussianBlur(depth, (5, 5), 0)
mean = cv2.blur(blur, (15, 15))
depth_feat = cv2.normalize(blur - mean, None, 0, 255, cv2.NORM_MINMAX)

# ===== 3. 强度增强（可关）=====
intensity = cv2.createCLAHE(2.0, (8, 8)).apply(intensity)

# ===== 4. 融合 =====
fused = cv2.addWeighted(intensity, 0.7, depth_feat, 0.3, 0)

cv2.imwrite(os.path.join(output_dir, name + ".png"), fused)


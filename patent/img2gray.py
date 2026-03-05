import cv2

# 请在此处指定图像路径
image_path = "./assets/main_new.png"  # 例如: "assets/example.jpg"

# 读取图像
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image from path:", image_path)
    exit(1)

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 生成输出路径（在原文件名后添加 '_gray'）
import os
base, ext = os.path.splitext(image_path)
output_path = base + '_gray' + ext

# 保存灰度图像
cv2.imwrite(output_path, gray_image)

print(f"Gray image saved as: {output_path}")

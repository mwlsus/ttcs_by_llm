import numpy as np
import cv2
import numpy as np
from skimage.feature import blob_log
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import io, color, morphology, measure
from skimage.filters import threshold_otsu  # 修正导入filters模块的方式
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
import argparse
import math
import matplotlib.gridspec as gridspec

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process some integers.')

# 添加 vpath 参数
parser.add_argument('--vpath', type=str, help='The path to the video file')

# 添加 name 参数
parser.add_argument('--fpath', type=str, help='The path to the video folder')

# 解析命令行参数
args = parser.parse_args()

# 打印参数值，或者根据参数执行相应的功能
print(f'Video path: {args.vpath}')
print(f'Folder path: {args.fpath}')
def get_output_path(input_path):
    # 获取输入文件的目录和文件名
    directory, filename = os.path.split(input_path)
    # 分离文件名和扩展名
    basename, extension = os.path.splitext(filename)
    # 构造输出文件的路径
    output_path = os.path.join(directory, f"{basename}_mask{extension}")
    return output_path


if __name__ == '__main__':
    folder_path = args.fpath
    video_path = args.vpath
    input_path = video_path

    # 获得输出文件路径
    output_path = get_output_path(input_path)
    folder_path = ''
    
    # 初始化背景减除器
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    # 打开视频文件或摄像头
    capture = cv2.VideoCapture(input_path)
    # 读取第一帧以获取视频尺寸
    ret, frame = capture.read()
    if not ret:
        print("无法读取视频")
        exit()
    
    height, width = frame.shape[:2]
    
    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height), False)
    
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重新定位到视频的开始
    current_frame = 0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = capture.read()
        if frame is None:
            break
    
        # 应用背景减除
        fgMask = backSub.apply(frame)
        blurred = cv2.GaussianBlur(fgMask, (3, 3), 0)
    
            # 写入帧到输出视频
        out.write(blurred)
        current_frame = current_frame + 1 
    # 释放资源
    capture.release()
    out.release()
    cv2.destroyAllWindows()
    print('mask ok')
    # 打开视频文件或摄像头
    capture = cv2.VideoCapture(output_path)
    # 读取第一帧以获取视频尺寸
    ret, frame = capture.read()
    if not ret:
        print("无法读取视频")
        exit()
    
    height, width = frame.shape[:2]
    
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重新定位到视频的开始
    current_frame = 0
    points = [] 
    while True:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = capture.read()
        if frame is None:
            break
    
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 使用blob_log方法检测斑点
        # 参数max_sigma设置检测的最大斑点的标准差，threshold是用于斑点检测的阈值 
        # 如果图像不是灰度图，转换为灰度图
        image_gray = color.rgb2gray(frame)
    
    
        # 二值化图像
        thresh = threshold_otsu(image_gray)  # 使用Otsu方法自动选择二值化阈值
        binary = image_gray > thresh
    
        # 标记连通区域
        label_image = measure.label(binary)
    
        # 创建一个全黑的图像用于绘制结果
        filtered_image = np.zeros_like(label_image)
        # 移除小于10x10的区域
        min_size = 100  # 设置最小大小
        max_size = 1000  # 设置最大大小
    
        for region in measure.regionprops(label_image):
            # 如果区域过小，则移除
            if region.area < min_size:
                # 将小区域的所有像素设为0（黑色）
                label_image[region.coords[:, 0], region.coords[:, 1]] = 0
            if region.area > max_size:
                label_image[region.coords[:, 0], region.coords[:, 1]] = 0
        # 保留大于等于10x10的白色区域
        #filtered_image = label_image > 0
    
        # 遍历所有区域，保留接近圆形的区域
        for region in measure.regionprops(label_image):
            # 计算圆形度
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            
            # 设置圆形度阈值，这里使用0.75作为示例，这个值可以根据需要调整
            if circularity > 0.5:
                # 将符合条件的区域填充到filtered_image中
                for coordinates in region.coords:
                    filtered_image[coordinates[0], coordinates[1]] = 1
                if region.centroid[1]>150:
                    points.append((current_frame,region.centroid))
        current_frame = current_frame + 1 
    x_coords = [point[1][0] for point in points]
    y_coords = [point[1][1] for point in points]
    # 假设data_x和data_y是你的数据列表
    data_x = x_coords
    data_y = y_coords
    fig, ax = plt.subplots()
    # 创建一个颜色映射
    cmap = plt.cm.Blues  # 选择蓝色系的colormap
    
    # 计算颜色值，根据索引比例变化
    colors = [cmap(i / max(len(data_x) - 1, 1)) for i in range(len(data_x))]
    
    # 绘制散点图
    ax.scatter( data_y, data_x,color=colors)
    
    # 在每个点上标注索引
    for i, (x, y) in enumerate(zip(data_x, data_y)):
        ax.text(y, x, str(i), color='red', fontsize=8, ha='right', va='bottom')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    # 显示图形
    plt.show()  
    sp =  int(input("请输入起始点"))
    ep = int(input("请输入结束点"))
    x_coords = [point[1][0] for point in points]
    y_coords = [point[1][1] for point in points]
    # 假设data_x和data_y是你的数据列表
    data_x = x_coords[sp:ep]
    data_y = y_coords[sp:ep]
    fig, ax = plt.subplots()
    # 创建一个颜色映射
    cmap = plt.cm.Blues  # 选择蓝色系的colormap
    
    # 计算颜色值，根据索引比例变化
    colors = [cmap(i / max(len(data_x) - 1, 1)) for i in range(len(data_x))]
    
    # 绘制散点图
    ax.scatter( data_y, data_x,color=colors)
    
    # 在每个点上标注索引
    for i, (x, y) in enumerate(zip(data_x, data_y)):
        ax.text(y, x, str(i), color='red', fontsize=8, ha='right', va='bottom')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    # 显示图形
    plt.show()  
    # 获取用户输入
    user_input = input("请输入要保留的点，以空格分隔: ")
    
    # 将输入字符串按空格分割，并转换为列表
    input_list = user_input.split()
    
    # 将列表中的每个元素转换为整数
    int_list = [int(item) for item in input_list]
    
    print("要保留的点是:", int_list)
    
    data_x =  [item for idx, item in enumerate(data_x) if idx in int_list]
    data_y =  [item for idx, item in enumerate(data_y) if idx in int_list]
    fig, ax = plt.subplots()
    # 创建一个颜色映射
    cmap = plt.cm.Blues  # 选择蓝色系的colormap
    
    # 计算颜色值，根据索引比例变化
    colors = [cmap(i / max(len(data_x) - 1, 1)) for i in range(len(data_x))]
    
    # 绘制散点图
    ax.scatter( data_y, data_x,color=colors)
    
    # 在每个点上标注索引
    for i, (x, y) in enumerate(zip(data_x, data_y)):
        ax.text(y, x, str(i), color='red', fontsize=8, ha='right', va='bottom')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([0,480])
    ax.set_xlim([0,960])
    ax.invert_yaxis()
    # 显示图形
    imgpath = r'{}\pptrace.jpg'.format(folder_path)
    plt.savefig(imgpath, dpi=300)
    plt.show()
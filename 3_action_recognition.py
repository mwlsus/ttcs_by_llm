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

def find_people(folder_path):
    # 获取文件夹中所有的.json文件
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # 按文件名排序
    sorted_json_files = sorted(json_files)
    json_now = os.path.join(folder_path,sorted_json_files[0])
    # 使用with语句打开文件，确保文件会被正确关闭
    with open(json_now, 'r') as file:
        # 直接读取文件内容为字符串
        json_str = file.read()
    data = json.loads(json_str)  # 这里假设json_str已经是一个有效的JSON字符串

    # 定义BODY_25模型的关键点连线
    pose_pairs = [
        [1, 8], [1, 2], [1, 5], [2, 3],
        [3, 4], [5, 6], [6, 7], [8, 9],
        [9, 10], [10, 11], [8, 12], [12, 13],
        [13, 14], [1, 0], [0, 15], [15, 17],
        [0, 16], [16, 18], [14, 19], [19, 20],
        [14, 21], [11, 22], [22, 23], [11, 24]
    ]
    
    # 绘图设置
    plt.figure(figsize=(10, 7))
    
    # 用于存储每个人的关键点信息
    people_keypoints = []
    
    # 遍历每个人
    for idx, person in enumerate(data['people'], start=1):
        # 获取pose_keypoints_2d数据
        keypoints = person['pose_keypoints_2d']
        # 分解关键点数据为x, y坐标和置信度c
        x = keypoints[0::3]
        y = keypoints[1::3]
        c = keypoints[2::3]
        
        # 存储关键点信息
        people_keypoints.append((x, y, c))
        
        # 绘制关键点
        for i in range(len(x)):
            if c[i] > 0.5:  # 根据需要调整置信度阈值
                plt.plot(x[i], y[i], 'o')
                plt.text(x[i], y[i], f'{idx}')  # 在关键点旁边标记编号
        
        # 绘制连线
        for pair in pose_pairs:
            partFrom = pair[0]
            partTo = pair[1]
            if c[partFrom] > 0.5 and c[partTo] > 0.5:  # 根据需要调整置信度阈值
                plt.plot([x[partFrom], x[partTo]], [y[partFrom], y[partTo]])
    
    # 反转y轴，以便图像看起来是正立的
    plt.gca().invert_yaxis()
    
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    
    # 用户输入
    user_input = int(input("请输入编号"))
    # 选中的人的关键点
    selected_keypoints = people_keypoints[user_input - 1]
    
    # 获取整条右臂的坐标（右肩，右臂，右前臂，右手腕）
    right_shoulder_select = (selected_keypoints[0][2], selected_keypoints[1][2])
    return right_shoulder_select


def openpose_data_get(folder_path,right_shoulder_select):
    # 获取文件夹中所有的.json文件
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    result = []
    # 按文件名排序
    sorted_json_files = sorted(json_files)
    for i in range(len(json_files)):
        json_now = os.path.join(folder_path,sorted_json_files[i])
        # 使用with语句打开文件，确保文件会被正确关闭
        with open(json_now, 'r') as file:
            # 直接读取文件内容为字符串
            json_str = file.read()
        data = json.loads(json_str)

        right_shoulder_point =right_shoulder_select  # 已知的右肩参考坐标
        
        # 初始化最小距离和最接近的人的数据
        min_distance = float('inf')
        closest_person_data = None
        
        # 遍历每个人
        for person in data['people']:
            keypoints = person['pose_keypoints_2d']
            # 分解关键点数据为x, y坐标
            x = keypoints[0::3]
            y = keypoints[1::3]
            
            # 获取此人的右肩（2）的坐标
            right_shoulder_x, right_shoulder_y = x[2], y[2]
            
            # 计算距离
            distance = np.sqrt((right_shoulder_x - right_shoulder_point[0])**2 + (right_shoulder_y - right_shoulder_point[1])**2)
            
            # 更新最小距离和最接近的人的数据
            if distance < min_distance:
                min_distance = distance
                closest_person_data = {
                    '2': (x[2], y[2]), # 右肩 
                    '3': (x[3], y[3]), # 右肘 
                    '4': (x[4], y[4]), # 右手
                    '5': (x[5], y[5]), # 左肩
                    '1': (x[1], y[1]), # 脖子
                    '8': (x[8], y[8])  # 裤裆
                }
        
        # 输出最接近的人的数据
        result.append(closest_person_data)
    return result

def calculate_angle(p1, p2, p3):
    """
    计算由三个点p1, p2, p3形成的角度，其中p2是顶点。
    每个点p都是一个(x, y)坐标对。
    """
    # 向量v1: p2到p1，向量v2: p2到p3
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # 计算向量的点积和向量的模长
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 计算两向量的夹角（弧度），然后转换为度
    angle = np.arccos(dot_prod / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle)
    return angle_deg
    
def triangle_centroid(point1, point2, point3):
    """
    计算并返回三角形重心的坐标。
    
    参数:
    - point1, point2, point3: 分别是三个顶点的坐标，每个点都是一个包含两个元素的元组，形式为(x, y)。
    
    返回值:
    - 重心的坐标，一个包含两个元素的元组，形式为(x, y)。
    """
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    Gx = (x1 + x2 + x3) / 3
    Gy = (y1 + y2 + y3) / 3
    return (Gx, Gy)

def calculate_s_angle(r_shoulder, l_shoulder):
    # 计算两点之间的斜率
    dy = r_shoulder[1] - l_shoulder[1]
    dx = r_shoulder[0] - l_shoulder[0]
    # 计算角度，atan2返回的是弧度
    angle = math.atan2(dy, dx)
    # 将弧度转换为度
    angle_degrees = math.degrees(angle)
    return angle_degrees

if __name__ == '__main__':
    folder_path = args.fpath
    video_path = args.vpath
    rss = find_people(folder_path)
    r = openpose_data_get(folder_path,rss)
    
    
    # 假设result包含了所有帧的数据
    result = r  # 这里应该是实际的帧数据列表
    num_c = 100
    print('frames:',len(r))
    if len(r) > num_c :
        num_c = num_c
        last_frames = result[-num_c:]
    else:
        num_c = len(r)
        last_frames = result
    
    
    # 存储计算出的角度
    angles = []
    distance_shoulders = []
    distance_elbows = []
    distance_wrists = []
    bcs = []
    shoulder_xs = []
    old_rs = None
    old_e = None
    old_w = None
    angles_s = []  # 存储每一帧的角度
    angle_differences = []  # 存储每一帧与上一帧之间角度的差值
    for frame in last_frames:
        # 提取右肩、右肘、右手的坐标
        r_shoulder = frame['2']
        l_shoulder = frame['5']
        hip = frame['8']
        elbow = frame['3']
        wrist = frame['4']
        
        # 计算角度并添加到列表中
        angle = calculate_angle(r_shoulder, elbow, wrist)
        angles.append(angle)
        if old_rs:
            distance_shoulder = np.sqrt((old_rs[0] - r_shoulder[0])**2 + (old_rs[1] - r_shoulder[1])**2)
            distance_shoulders.append(distance_shoulder)
        if old_e:
            distance_elbow = np.sqrt((old_e[0] - elbow[0])**2 + (old_e[1] - elbow[1])**2)
            distance_elbows.append(distance_elbow)
        if old_w:
            distance_wrist = np.sqrt((old_w[0] - wrist[0])**2 + (old_w[1] - wrist[1])**2)
            distance_wrists.append(distance_wrist)
        old_rs = r_shoulder
        old_e = elbow
        old_w = wrist
        bc = triangle_centroid(r_shoulder,l_shoulder,hip)
        bcs.append(bc)
        shoulder_xs.append(r_shoulder[0])
        angle = calculate_s_angle(r_shoulder, l_shoulder)
        angles_s.append(angle)
    # 计算相邻两帧之间角度的差值
    for i in range(1, len(angles_s)):
        angle_difference = np.abs(angles_s[i] - angles_s[i-1])
        angle_differences.append(np.min((angle_difference,360-angle_difference)))
    bcx, bcy = zip(*bcs)
    min_value = min(bcx)
    bcx = [x - min_value for x in bcx]
    min_value = min(bcy)
    bcy = [x - min_value for x in bcy]
    min_value = min(shoulder_xs)
    shoulder_xs = [x - min_value for x in shoulder_xs]
    
    # 绘图
    # 创建画布，设置为正方形，例如尺寸为6x18英寸以保持整体为正方形
    plt.figure(figsize=(6, 6))
    # 添加第一个子图
    ax1 = plt.subplot(3, 1, 1)  # (rows, columns, panel number)
    plt.plot(distance_shoulders)
    ax1.set_ylim([0, 5])  # 设置y轴尺度为0到5
    # 添加第二个子图
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(distance_elbows)
    ax2.set_ylim([0, 10])  # 设置y轴尺度为0到10
    # 添加第三个子图
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(distance_wrists)
    ax3.set_ylim([0, 50])  # 设置y轴尺度为-2到6
    plt.tight_layout()
    imgpath = os.path.join(folder_path,'arm_vector.jpg')
    plt.savefig(imgpath, dpi=300)
    
    
    plt.figure(figsize=(6,4))
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(angles)
    ax1.set_ylim([70, 150])  # 设置y轴尺度为0到5
    plt.ylabel('Arm Angle (Degrees)')
    imgpath = os.path.join(folder_path,'arm_angle.jpg')
    plt.savefig(imgpath, dpi=300)
    
    
    plt.figure(figsize=(6,4))
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(bcx)
    ax1.set_ylim([0, 10])  # 设置y轴尺度为0到5
    imgpath = os.path.join(folder_path,'bcx.jpg')
    plt.savefig(imgpath, dpi=300)
    
    
    plt.figure(figsize=(6,4))
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(bcy)
    ax1.set_ylim([0, 15])  # 设置y轴尺度为0到5
    imgpath = os.path.join(folder_path,'bcy.jpg')
    plt.savefig(imgpath, dpi=300)
    
    
    plt.figure(figsize=(6,4))
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(shoulder_xs)
    ax1.set_ylim([0, 15])  # 设置y轴尺度为0到5
    imgpath = os.path.join(folder_path,'shoulder_x.jpg')
    plt.savefig(imgpath, dpi=300)
    
    
    plt.figure(figsize=(6,4))
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(angle_differences)
    ax1.set_ylim([0, 20])  # 设置y轴尺度为0到5
    imgpath = os.path.join(folder_path,'shoulder_shake.jpg')
    plt.savefig(imgpath, dpi=300)
    
    
    # 创建一个图figure
    plt.figure(figsize=(10, 8))
    
    # 创建一个两行三列的布局
    gs = gridspec.GridSpec(2, 3)
    
    # 第一行第一列的位置细分为三个子图
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0])
    
    
    
    ax = plt.subplot(gs0[0])
    ax.plot(distance_shoulders)  # 仅作示例，您可以根据需要绘制不同的图形
    ax.xaxis.set_visible(False)
    ax.set_ylim([0, 5])  # 设置y轴尺度为0到5
    ax.set_title('Arm distance')
    ax = plt.subplot(gs0[1])
    ax.plot(distance_elbows)  # 仅作示例，您可以根据需要绘制不同的图形
    ax.xaxis.set_visible(False)
    ax.set_ylim([0, 10])  # 设置y轴尺度为0到5
    ax = plt.subplot(gs0[2])
    ax.plot(distance_wrists)  # 仅作示例，您可以根据需要绘制不同的图形
    ax.set_ylim([0, 50])  # 设置y轴尺度为0到5
    
    # 绘制其他五个正方形图
    # 注意：由于第一个位置已被细分，索引调整为从1开始
    ax = plt.subplot(gs[1])
    ax.plot(bcx)
    ax.set_ylim([0, 30])
    ax.set_title('Center of gravity X') 
    ax = plt.subplot(gs[2])
    ax.plot(bcy)
    ax.set_ylim([0, 10])
    ax.set_title('Center of gravity Y')
    
    ax = plt.subplot(gs[3])
    ax.plot(angles)
    ax.set_ylim([70, 150]) 
    ax.set_title('Arm angle')
    
    ax = plt.subplot(gs[4])
    ax.plot(shoulder_xs)
    ax.set_ylim([0, 30]) 
    ax.set_title('Shoulder displacement')
    
    ax = plt.subplot(gs[5])
    ax.plot(angle_differences)
    ax.set_ylim([0, 20]) 
    ax.set_title('Shoulder shake')
    
    imgpath = os.path.join(folder_path,'all.jpg')
    plt.savefig(imgpath, dpi=300)
    # 第75到第100个数据点（Python中索引从0开始）
    subset = distance_wrists[-25:]
    
    # 在子集中找到最大值
    max_value = max(subset)
    
    # 找出最大值在原始列表中的索引
    max_index = distance_wrists.index(max_value)
    
    #print("最大值的索引是:", max_index)
    daoshu = len(distance_shoulders) - max_index
    kf = result[-daoshu]
    
    
    # 所需的坐标点和帧
    x, y = kf['4']# 这里是示例坐标，根据你的需要修改
    x = int(x)
    y = int(y)
    frame_number = len(result) -daoshu   # 示例帧数，根据你的需要修改
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 设置视频帧号
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 读取指定帧
    success, frame = cap.read()
    
    if success:
        # 获取图像的高度和宽度
        height, width, _ = frame.shape
        
        # 计算裁剪框的左上角坐标
        x1 = max(0, x - 50)
        y1 = max(0, y - 50)
        
        # 确保裁剪框不会超出图像边界
        x2 = min(width, x + 50)
        y2 = min(height, y + 50)
        
        # 裁剪图像
        cropped_img = frame[y1:y2, x1:x2]
        imgpath = os.path.join(folder_path,'paddle.jpg')
        # 保存图像
        cv2.imwrite(imgpath, cropped_img)
    
        
        # 提取所有的x和y坐标
        x_coords = [coord[0] for coord in kf.values()]
        y_coords = [coord[1] for coord in kf.values()]
        
        # 找到最小和最大的x,y坐标
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 调整边界
        x1 = max(0, int(min_x - 50))
        y1 = max(0, int(min_y - 300))
        x2 = min(width, int(max_x + 50))
        y2 = min(height, int(max_y + 500))
        
        cropped_img = frame[y1:y2, x1:x2]
        imgpath = os.path.join(folder_path,'people.jpg')
        # 保存图像
        cv2.imwrite(imgpath, cropped_img)
    
    
    else:
        print("无法读取指定帧")
    
    # 释放视频对象
    cap.release()
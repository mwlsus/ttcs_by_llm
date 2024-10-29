import cv2
import argparse
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process some integers.')

# 添加 vpath 参数
parser.add_argument('--vpath', type=str, help='The path to the video file')

# 添加 name 参数
parser.add_argument('--name', type=str, help='The name identifier for the video')

# 解析命令行参数
args = parser.parse_args()

# 打印参数值，或者根据参数执行相应的功能
print(f'Video path: {args.vpath}')
print(f'Name: {args.name}')

if __name__ == '__main__':
    # 初始化起点和终点列表
    start_points = []
    end_points = []
    # 打开视频文件
    cap = cv2.VideoCapture(vpath)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    current_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'total frames : {total_frames}')
    while True:
        # 读取当前帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        # 如果读取帧失败，跳过
        if not ret:
            print("Error: Could not read frame.")
            break
        # 调整帧大小为原视频的一半仅用于显示
        frame_half_size = cv2.resize(frame, None, fx=0.3, fy=0.3)

        # 显示调整大小的帧
        cv2.imshow('Video', frame_half_size)

        # 键盘输入
        key = cv2.waitKey(0) & 0xFF

        if key == ord('1'):
            # 前进一帧
            current_frame = min(current_frame + 1, total_frames - 1)
        elif key == ord('2'):
            # 后退一帧
            current_frame = max(current_frame - 1, 0)

        if key == ord('3'):
            # 前进4帧
            current_frame = min(current_frame + 4, total_frames - 1)
        elif key == ord('4'):
            # 后退4帧
            current_frame = max(current_frame - 4, 0)

        if key == ord('5'):
            # 前进30帧
            current_frame = min(current_frame + 30, total_frames - 1)
        elif key == ord('6'):
            # 后退30帧
            current_frame = max(current_frame - 30, 0)

        elif key == ord('s'):
            # 保存起点
            start_points.append(current_frame)
            print(f"Start point at frame {current_frame} saved.")
        elif key == ord('e'):
            # 保存终点
            end_points.append(current_frame)
            print(f"End point at frame {current_frame} saved.")

        elif key == ord('n'):
            # 查看当前frame number
        
            print(f"Now at frame {current_frame}.")

        elif key == 27:  # ESC key
            # 退出
            print("Exiting and processing video cuts...")
            break

    cv2.destroyAllWindows()
    cap.release()

    # 确保起点和终点数量匹配
    if len(start_points) != len(end_points):
        print("Error: Start points and end points count does not match.")
    else:
        # 剪辑视频
        for i, (start, end) in enumerate(zip(start_points, end_points)):
            cap = cv2.VideoCapture(vpath)
            # 设置视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(f'{name}_{i}.avi', fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))

            # 读取并写入视频片段
            for frame_num in range(start, end + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
            
            cap.release()
            out.release()
            print(f"Video cut {i} from frame {start} to {end} saved as output_{i}.avi")
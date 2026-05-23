import requests
import json

# 配置
URL = "http://192.168.4.1"
buf = b''

print("✅ 已连接，同时接收视频 + GY85 数据...")

# 单线程接收
stream = requests.get(URL, stream=True, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)

for chunk in stream.iter_content(4096):
    buf += chunk

    # 找一帧的开始和结束
    s = buf.find(b'--123456789000000000000987654321')
    if s == -1:
        continue

    # 截取一帧内容
    frame = buf[:s]
    buf = buf[s + 67:]

    # 1. 提取 IMU 数据
    imu_start = frame.find(b'{"ax":')
    imu_end = frame.find(b'}', imu_start) + 1
    if imu_start > 0 and imu_end > 0:
        imu_str = frame[imu_start:imu_end].decode()
        imu = json.loads(imu_str)
        print(f"加速度: {imu['ax']:.4f} {imu['ay']:.4f} {imu['az']:.4f} | 陀螺仪: {imu['gx']:.4f} {imu['gy']:.4f} {imu['gz']:.4f}")

    # 2. 提取图片
    jpg_s = frame.find(b'\xff\xd8')
    jpg_e = frame.find(b'\xff\xd9')
    if jpg_s != -1 and jpg_e != -1:
        jpg = frame[jpg_s:jpg_e+2]
        with open("frame.jpg", "wb") as f:
            f.write(jpg)
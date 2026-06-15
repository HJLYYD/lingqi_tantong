import requests
import json
import time
import signal
import sys

# 配置
URL = "http://192.168.4.1"
RETRY_DELAY = 3   # 重试间隔（秒）
MAX_RETRIES = 5   # 最大重试次数

running = True

def sig_handler(sig, frame):
    global running
    running = False
    print("\n🛑 正在断开...")

signal.signal(signal.SIGINT, sig_handler)

def connect_stream():
    """连接 MJPEG 流，失败时自动重试"""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            print(f"🔗 正在连接 {URL} ..." + (f" (重试 {retries}/{MAX_RETRIES})" if retries else ""))
            # connect timeout=5s, read timeout=None (无限流式读)
            stream = requests.get(
                URL,
                stream=True,
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Connection': 'close',        # 每次请求后关闭连接
                    'Cache-Control': 'no-cache',
                },
                timeout=(5, None),                # (connect_timeout, read_timeout)
            )
            if stream.status_code == 200:
                print("✅ 已连接，同时接收视频 + GY85 数据...")
                return stream
            else:
                print(f"❌ HTTP {stream.status_code}")
        except requests.exceptions.ReadTimeout:
            print(f"⚠️ 读超时，{RETRY_DELAY}s 后重试...")
        except requests.exceptions.ConnectionError as e:
            print(f"⚠️ 连接失败: {e}")
            print(f"   请确认: 1) 已连上 ESP32 热点  2) ESP32 正在运行")
        except Exception as e:
            print(f"⚠️ 未知错误: {e}")

        retries += 1
        if retries < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return None

# ---- 主循环 ----
while running:
    stream = connect_stream()
    if stream is None:
        print("❌ 超过最大重试次数，退出")
        sys.exit(1)

    buf = b''
    total_bytes = 0
    frame_count = 0
    imu_count = 0
    jpg_count = 0
    first_chunk = True

    try:
        for chunk in stream.iter_content(4096):
            if not running:
                break

            total_bytes += len(chunk)
            buf += chunk

            # 首次收到数据时打印详细信息
            if first_chunk and len(buf) > 0:
                first_chunk = False
                print(f"📦 首次收到数据: {len(chunk)} 字节 (累计缓冲 {len(buf)} 字节)")
                # 打印前 200 字节的 hex/ascii
                head = buf[:200]
                print(f"   [HEX] {head[:100].hex()}")
                if len(head) > 100:
                    print(f"   [HEX] {head[100:200].hex()}")

            # 找边界
            s = buf.find(b'--123456789000000000000987654321')
            if s == -1:
                # 每 5 秒打印一次心跳（防止看起来像卡死）
                if total_bytes % 20480 < 4096 and total_bytes > 0:
                    pass  # 太频繁了，略过
                continue

            # 截取一帧
            frame = buf[:s]
            frame_len = len(frame)
            buf = buf[s + 67:]
            frame_count += 1

            # 帧内搜索
            imu_start = frame.find(b'{"ax":')
            imu_end = frame.find(b'}', imu_start) + 1 if imu_start >= 0 else -1
            jpg_s = frame.find(b'\xff\xd8')
            jpg_e = frame.find(b'\xff\xd9')

            # 第1帧 + 每10帧打印总结
            if frame_count == 1 or frame_count % 10 == 0:
                has_imu = "✓" if (imu_start >= 0 and imu_end > 0) else "✗"
                has_jpg = "✓" if (jpg_s != -1 and jpg_e != -1) else "✗"
                print(f"[RX] 第{frame_count}帧 | 帧长={frame_len}B | IMU={has_imu} JPEG={has_jpg} | 累计={total_bytes}B")

            # 1. 提取 IMU 数据（imu_start 可能为 0，去掉 > 0 限制）
            if imu_start >= 0 and imu_end > 0:
                try:
                    imu_str = frame[imu_start:imu_end].decode()
                    imu = json.loads(imu_str)
                    imu_count += 1
                    print(f"   IMU | ax={imu['ax']:.4f} ay={imu['ay']:.4f} az={imu['az']:.4f} | gx={imu['gx']:.4f} gy={imu['gy']:.4f} gz={imu['gz']:.4f}")
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                    print(f"   ⚠️ IMU 解析失败: {e} | raw={frame[imu_start:imu_end][:80]}")

            # 2. 提取图片
            if jpg_s != -1 and jpg_e != -1:
                jpg = frame[jpg_s:jpg_e+2]
                with open("frame.jpg", "wb") as f:
                    f.write(jpg)
                jpg_count += 1

    except requests.exceptions.ChunkedEncodingError as e:
        print(f"⚠️ 流中断 (ChunkedEncodingError): {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"⚠️ 连接断开 (ConnectionError): {e}")
    except Exception as e:
        print(f"⚠️ 流异常: {type(e).__name__}: {e}")

    finally:
        stream.close()
        print(f"📴 连接关闭 | 累计接收={total_bytes}B | 帧={frame_count} | IMU={imu_count} | JPEG={jpg_count}")

    if running:
        print(f"⏳ {RETRY_DELAY}s 后自动重连...")
        time.sleep(RETRY_DELAY)

print("👋 退出")
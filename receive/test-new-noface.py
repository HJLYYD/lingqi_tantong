import numpy as np
import requests
import os

# 忽略所有警告
os.environ['QT_LOGGING_RULES'] = 'qt.fontdatabase.debug=false'

STREAM_URL = "http://192.168.4.1"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

stream = requests.get(STREAM_URL, stream=True, headers=HEADERS, timeout=10)
bytes_data = b''

print("✅ 流已连接！打开文件夹里的 frame.jpg 查看画面")

while True:
    for chunk in stream.iter_content(4096):
        bytes_data += chunk
        start = bytes_data.find(b'\xff\xd8')
        end = bytes_data.find(b'\xff\xd9')
        if start != -1 and end != -1:
            jpg = bytes_data[start:end+2]
            bytes_data = bytes_data[end+2:]
            # 🔥 核心：直接保存图片到本地
            with open("frame.jpg", "wb") as f:
                f.write(jpg)
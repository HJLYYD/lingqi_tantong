#!/usr/bin/env python3
"""CoAP 客户端：/stream 视频 + /imu 1Hz + MG996R 舵机 GUI 控制"""

import json
import socket
import struct
import sys
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

COAP_HOST = "192.168.4.1"
COAP_PORT = 5683
BLOCK_SZX = 6
STREAM_TIMEOUT = 6.0
SERVO_TIMEOUT = 1.0
IMU_INTERVAL = 1.0

STREAM_TOKEN = b"\xca\xfe\xba\xbe"
IMU_TOKEN = b"\xde\xad\xbe\xef"
SERVO_TOKEN = b"\xbe\xef\xca\xfe"

SERVO_ANGLE_LEFT = 0
SERVO_ANGLE_STOP = 90
SERVO_ANGLE_RIGHT = 180

ROOT_DIR = Path(__file__).resolve().parent
FRAME_PATH = ROOT_DIR / "frame.jpg"


def block_size(szx: int) -> int:
    return 16 << szx


def encode_option(delta: int, value: bytes) -> bytes:
    out = bytearray()
    d, v = delta, len(value)
    while True:
        if d <= 12 and v <= 12:
            out.append((d << 4) | v)
            break
        if d <= 12:
            out.append((d << 4) | 13)
            out.append(v - 13)
            break
        if d <= 255:
            out.append(13 << 4)
            out.append(d - 13)
            d = 0
            continue
        return b""
    out.extend(value)
    return bytes(out)


def build_get(path: str, msg_id: int, token: bytes) -> bytes:
    parts = path.strip("/").split("/") if path.strip("/") else []
    opts = bytearray()
    prev = 0
    for seg in parts:
        seg_b = seg.encode()
        opts.extend(encode_option(11 - prev, seg_b))
        prev = 11
    hdr = bytes([(1 << 6) | (0 << 4) | len(token), 0x01]) + struct.pack(">H", msg_id) + token
    payload_marker = b"\xff" if not opts else b""
    return hdr + bytes(opts) + payload_marker


def build_put(path: str, msg_id: int, token: bytes, payload: bytes) -> bytes:
    parts = path.strip("/").split("/") if path.strip("/") else []
    opts = bytearray()
    prev = 0
    for seg in parts:
        seg_b = seg.encode()
        opts.extend(encode_option(11 - prev, seg_b))
        prev = 11
    hdr = bytes([(1 << 6) | (0 << 4) | len(token), 0x03]) + struct.pack(">H", msg_id) + token
    return hdr + bytes(opts) + b"\xff" + payload


def parse_packet(data: bytes):
    token_len = data[0] & 0x0F
    pos = 4 + token_len
    token = data[4:4 + token_len]
    opts = {}
    payload = b""
    opt_num = 0
    while pos < len(data):
        if data[pos] == 0xFF:
            pos += 1
            payload = data[pos:]
            break
        hdr = data[pos]
        pos += 1
        delta = (hdr >> 4) & 0x0F
        olen = hdr & 0x0F
        if delta == 13:
            delta = data[pos] + 13
            pos += 1
        elif delta == 14:
            delta = (data[pos] << 8 | data[pos + 1]) + 269
            pos += 2
        if olen == 13:
            olen = data[pos] + 13
            pos += 1
        elif olen == 14:
            olen = (data[pos] << 8 | data[pos + 1]) + 269
            pos += 2
        opt_num += delta
        val = data[pos:pos + olen]
        pos += olen
        opts[opt_num] = val
    return token, opts, payload


def parse_block2(val: bytes):
    n = int.from_bytes(val, "big")
    return n >> 4, bool(n & 0x08), n & 0x07


def receive_jpeg_frames(sock: socket.socket, token: bytes):
    blocks = {}
    expected_etag = None
    expected_total = None
    expected_szx = BLOCK_SZX
    next_block = 0
    frame_count = 0
    fps_t0 = time.monotonic()

    while True:
        data, _ = sock.recvfrom(65536)
        pkt_token, opts, chunk = parse_packet(data)
        if pkt_token != token or 23 not in opts:
            continue

        num, more, szx = parse_block2(opts[23])

        if num == 0:
            etag = int.from_bytes(opts[4], "big") if 4 in opts else 0
            total = int.from_bytes(opts[28], "big") if 28 in opts else None
            blocks = {0: chunk}
            expected_etag = etag
            expected_total = total
            expected_szx = szx
            next_block = 1
        elif expected_etag is not None and num == next_block:
            blocks[num] = chunk
            next_block += 1
            szx = expected_szx
        else:
            continue

        if more and len(chunk) != block_size(szx):
            continue

        if not more:
            if expected_total:
                if len(blocks) * block_size(szx) < expected_total and len(blocks) < (
                    (expected_total + block_size(szx) - 1) // block_size(szx)
                ):
                    continue
                jpg = bytearray(expected_total)
                offset = 0
                for i in sorted(blocks):
                    part = blocks[i]
                    jpg[offset : offset + len(part)] = part
                    offset += len(part)
                jpg = bytes(jpg[:offset])
            else:
                jpg = b"".join(blocks[i] for i in sorted(blocks))

            if expected_total and len(jpg) != expected_total:
                blocks = {}
                continue
            if len(jpg) < 2 or jpg[:2] != b"\xff\xd8":
                blocks = {}
                continue

            frame_count += 1
            now = time.monotonic()
            if now - fps_t0 >= 2.0:
                fps = frame_count / (now - fps_t0)
                print(f"视频 FPS: {fps:.1f}  帧大小: {len(jpg)//1024}KB", file=sys.stderr)
                frame_count = 0
                fps_t0 = now

            yield jpg
            blocks = {}
            expected_etag = None
            expected_total = None
            next_block = 0


def imu_poll_loop(host: str) -> None:
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)
            msg_id = 1
            while True:
                req = build_get("imu", msg_id, IMU_TOKEN)
                sock.sendto(req, (host, COAP_PORT))
                data, _ = sock.recvfrom(1024)
                _, opts, payload = parse_packet(data)
                if payload:
                    imu = json.loads(payload.decode())
                    print(
                        f"加速度: {imu['ax']:.4f} {imu['ay']:.4f} {imu['az']:.4f} | "
                        f"陀螺仪: {imu['gx']:.4f} {imu['gy']:.4f} {imu['gz']:.4f}"
                    )
                msg_id = (msg_id + 1) & 0xFFFF
                time.sleep(IMU_INTERVAL)
        except Exception:
            time.sleep(0.5)


def stream_loop(host: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    sock.settimeout(STREAM_TIMEOUT)
    stream_msg_id = 1
    sock.sendto(build_get("stream", stream_msg_id, STREAM_TOKEN), (host, COAP_PORT))

    while True:
        try:
            for jpg in receive_jpeg_frames(sock, STREAM_TOKEN):
                with open(FRAME_PATH, "wb") as f:
                    f.write(jpg)
        except socket.timeout:
            print("视频超时，重新订阅...", file=sys.stderr)
            stream_msg_id = (stream_msg_id + 1) & 0xFFFF
            sock.sendto(build_get("stream", stream_msg_id, STREAM_TOKEN), (host, COAP_PORT))
        except Exception as e:
            print(f"视频错误: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(0.3)
            stream_msg_id = (stream_msg_id + 1) & 0xFFFF
            sock.sendto(build_get("stream", stream_msg_id, STREAM_TOKEN), (host, COAP_PORT))


class ServoClient:
    def __init__(self, host: str = COAP_HOST, port: int = COAP_PORT):
        self.host = host
        self.port = port
        self._msg_id = 100
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(SERVO_TIMEOUT)

    def close(self) -> None:
        self._sock.close()

    def send_angle(self, angle: int) -> None:
        with self._lock:
            payload = json.dumps({"angle": angle}).encode()
            req = build_put("servo", self._msg_id, SERVO_TOKEN, payload)
            self._msg_id = (self._msg_id + 1) & 0xFFFF
        try:
            self._sock.sendto(req, (self.host, self.port))
        except OSError as e:
            print(f"舵机发送失败: {e}", file=sys.stderr)

    def stop(self) -> None:
        self.send_angle(SERVO_ANGLE_STOP)

    def left(self) -> None:
        self.send_angle(SERVO_ANGLE_LEFT)

    def right(self) -> None:
        self.send_angle(SERVO_ANGLE_RIGHT)


class ServoControlApp:
    def __init__(self, root: tk.Tk, client: ServoClient, host: str = COAP_HOST):
        self.root = root
        self.client = client
        self.host = host
        self._left_pressed = False
        self._right_pressed = False

        root.title("OV3660 + IMU + 舵机控制")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        title_font = tkfont.Font(size=16, weight="bold")
        btn_font = tkfont.Font(size=28, weight="bold")
        status_font = tkfont.Font(size=12)

        frame = tk.Frame(root, padx=24, pady=20)
        frame.pack()

        tk.Label(frame, text="按住旋转，松开停止", font=title_font).grid(
            row=0, column=0, columnspan=2, pady=(0, 16)
        )

        self.left_btn = tk.Button(
            frame,
            text="左",
            width=6,
            height=2,
            font=btn_font,
            bg="#4a90d9",
            fg="white",
            activebackground="#357abd",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
        )
        self.left_btn.grid(row=1, column=0, padx=(0, 12))

        self.right_btn = tk.Button(
            frame,
            text="右",
            width=6,
            height=2,
            font=btn_font,
            bg="#d94a4a",
            fg="white",
            activebackground="#bd3535",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
        )
        self.right_btn.grid(row=1, column=1, padx=(12, 0))

        self.status_var = tk.StringVar(value="状态: 停止")
        tk.Label(frame, textvariable=self.status_var, font=status_font, fg="#444").grid(
            row=2, column=0, columnspan=2, pady=(16, 0)
        )

        tk.Label(
            frame,
            text=f"CoAP: {self.host}:{COAP_PORT}/servo",
            font=tkfont.Font(size=10),
            fg="#888",
        ).grid(row=3, column=0, columnspan=2, pady=(8, 0))

        for btn, side in ((self.left_btn, "left"), (self.right_btn, "right")):
            btn.bind("<ButtonPress-1>", lambda e, s=side: self.on_press(s))
            btn.bind("<ButtonRelease-1>", lambda e, s=side: self.on_release(s))
            btn.bind("<Leave>", lambda e, s=side: self.on_leave(s))

        root.bind("<ButtonRelease-1>", self.on_global_release)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def on_press(self, side: str) -> None:
        if side == "left":
            self._left_pressed = True
            self._right_pressed = False
            self.client.left()
            self.set_status("状态: 逆时针旋转")
        else:
            self._right_pressed = True
            self._left_pressed = False
            self.client.right()
            self.set_status("状态: 顺时针旋转")

    def on_release(self, side: str) -> None:
        if side == "left" and self._left_pressed:
            self._left_pressed = False
            self.client.stop()
            self.set_status("状态: 停止")
        elif side == "right" and self._right_pressed:
            self._right_pressed = False
            self.client.stop()
            self.set_status("状态: 停止")

    def on_leave(self, side: str) -> None:
        if side == "left" and self._left_pressed:
            self._left_pressed = False
            self.client.stop()
            self.set_status("状态: 停止")
        elif side == "right" and self._right_pressed:
            self._right_pressed = False
            self.client.stop()
            self.set_status("状态: 停止")

    def on_global_release(self, _event=None) -> None:
        if self._left_pressed or self._right_pressed:
            self._left_pressed = False
            self._right_pressed = False
            self.client.stop()
            self.set_status("状态: 停止")

    def on_close(self) -> None:
        self.client.stop()
        self.client.close()
        self.root.destroy()


def main() -> None:
    host = COAP_HOST
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            print("用法: ov-imu-pwm.py [ESP32_IP]")
            print("  无参数: 打开舵机控制窗口，同时订阅视频流与 IMU")
            print("  带 IP:  指定 ESP32 热点 IP (默认 192.168.4.1)")
            print()
            print(f"  视频帧保存至 {FRAME_PATH}")
            print(f"  IMU 数据以 {1/IMU_INTERVAL:.0f}Hz 打印到终端")
            return
        host = sys.argv[1]

    print(f"视频流: coap://{host}:{COAP_PORT}/stream")
    print(f"IMU 通道: coap://{host}:{COAP_PORT}/imu ({1/IMU_INTERVAL:.0f}Hz)")
    print(f"舵机控制: coap://{host}:{COAP_PORT}/servo")

    threading.Thread(target=imu_poll_loop, args=(host,), daemon=True).start()
    threading.Thread(target=stream_loop, args=(host,), daemon=True).start()

    root = tk.Tk()
    client = ServoClient(host=host)
    ServoControlApp(root, client, host=host)
    root.mainloop()


if __name__ == "__main__":
    main()

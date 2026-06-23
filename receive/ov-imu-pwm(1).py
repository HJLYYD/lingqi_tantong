#!/usr/bin/env python3
"""CoAP 客户端：/stream 视频 + /imu 1Hz + MG996R 舵机 GUI 控制"""

import json
import os
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
PACKET_TIMEOUT = 1.0
KEEPALIVE_INTERVAL = 8.0
IDLE_RESUBSCRIBE = 25.0
FRAME_STALE = 4.0
SERVO_TIMEOUT = 1.0
IMU_INTERVAL = 1.0

STREAM_TOKEN = b"\xca\xfe\xba\xbe"
IMU_TOKEN = b"\xde\xad\xbe\xef"
SERVO_TOKEN = b"\xbe\xef\xca\xfe"

SERVO_ANGLE_LEFT = 0
SERVO_ANGLE_STOP = 90
SERVO_ANGLE_RIGHT = 180

SERVO_H = 0  # GPIO40 左右
SERVO_V = 1  # GPIO41 上下

ROOT_DIR = Path(__file__).resolve().parent
FRAME_PATH = ROOT_DIR / "frame.jpg"
FRAME_TMP_PATH = ROOT_DIR / "frame.tmp"


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


class JpegStreamAssembler:
    """按 ETag + Block2 重组 JPEG，支持乱序块、丢弃超时半成品帧。"""

    def __init__(self, token: bytes):
        self.token = token
        self.reset()

    def reset(self) -> None:
        self.blocks: dict[int, bytes] = {}
        self.etag: Optional[int] = None
        self.total: Optional[int] = None
        self.szx = BLOCK_SZX
        self.started_at = 0.0

    def feed(self, data: bytes) -> Optional[bytes]:
        if self.started_at and time.monotonic() - self.started_at > FRAME_STALE:
            self.reset()

        pkt_token, opts, chunk = parse_packet(data)
        if pkt_token != self.token or 23 not in opts:
            return None

        num, more, szx = parse_block2(opts[23])
        etag = int.from_bytes(opts[4], "big") if 4 in opts else None

        if num == 0:
            self.blocks = {0: chunk}
            self.etag = etag
            self.total = int.from_bytes(opts[28], "big") if 28 in opts else None
            self.szx = szx
            self.started_at = time.monotonic()
        elif self.etag is not None and etag in (None, self.etag):
            self.blocks[num] = chunk
        else:
            return None

        blk_sz = block_size(self.szx)
        if more and len(chunk) != blk_sz:
            return None

        if more:
            return None

        if self.total is not None:
            need = (self.total + blk_sz - 1) // blk_sz
            if len(self.blocks) < need:
                return None
            jpg = b"".join(self.blocks[i] for i in range(need))
        else:
            jpg = b"".join(self.blocks[i] for i in sorted(self.blocks))

        if self.total is not None and len(jpg) != self.total:
            self.reset()
            return None
        if len(jpg) < 2 or jpg[:2] != b"\xff\xd8":
            self.reset()
            return None

        self.reset()
        return jpg


def stream_loop(host: str) -> None:
    msg_id = 1

    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            sock.settimeout(PACKET_TIMEOUT)
            assembler = JpegStreamAssembler(STREAM_TOKEN)
            last_packet = time.monotonic()
            last_keepalive = 0.0

            sock.sendto(build_get("stream", msg_id, STREAM_TOKEN), (host, COAP_PORT))
            msg_id = (msg_id + 1) & 0xFFFF
            last_packet = time.monotonic()

            while True:
                try:
                    data, _ = sock.recvfrom(65536)
                    last_packet = time.monotonic()
                except socket.timeout:
                    now = time.monotonic()
                    idle = now - last_packet
                    if idle >= IDLE_RESUBSCRIBE:
                        break
                    if idle >= KEEPALIVE_INTERVAL and now - last_keepalive >= KEEPALIVE_INTERVAL:
                        sock.sendto(build_get("stream", msg_id, STREAM_TOKEN), (host, COAP_PORT))
                        msg_id = (msg_id + 1) & 0xFFFF
                        last_keepalive = now
                    continue

                jpg = assembler.feed(data)
                if jpg:
                    FRAME_TMP_PATH.write_bytes(jpg)
                    os.replace(FRAME_TMP_PATH, FRAME_PATH)
        except OSError:
            time.sleep(0.5)
        finally:
            sock.close()

        time.sleep(0.3)
        msg_id = (msg_id + 1) & 0xFFFF


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

    def send_angle(self, angle: int, servo: int = SERVO_H) -> None:
        with self._lock:
            payload = json.dumps({"servo": servo, "angle": angle}).encode()
            req = build_put("servo", self._msg_id, SERVO_TOKEN, payload)
            self._msg_id = (self._msg_id + 1) & 0xFFFF
        try:
            self._sock.sendto(req, (self.host, self.port))
        except OSError as e:
            print(f"舵机发送失败: {e}", file=sys.stderr)

    def stop(self, servo: int = SERVO_H) -> None:
        self.send_angle(SERVO_ANGLE_STOP, servo)

    def stop_all(self) -> None:
        self.stop(SERVO_H)
        self.stop(SERVO_V)

    def left(self) -> None:
        self.send_angle(SERVO_ANGLE_LEFT, SERVO_H)

    def right(self) -> None:
        self.send_angle(SERVO_ANGLE_RIGHT, SERVO_H)

    def up(self) -> None:
        self.send_angle(SERVO_ANGLE_LEFT, SERVO_V)

    def down(self) -> None:
        self.send_angle(SERVO_ANGLE_RIGHT, SERVO_V)


class ServoControlApp:
    def __init__(self, root: tk.Tk, client: ServoClient, host: str = COAP_HOST):
        self.root = root
        self.client = client
        self.host = host
        self._pressed: set[str] = set()

        root.title("OV3660 + IMU + 双舵机控制")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        title_font = tkfont.Font(size=16, weight="bold")
        btn_font = tkfont.Font(size=28, weight="bold")
        status_font = tkfont.Font(size=12)

        frame = tk.Frame(root, padx=24, pady=20)
        frame.pack()

        tk.Label(frame, text="按住旋转，松开停止", font=title_font).grid(
            row=0, column=0, columnspan=3, pady=(0, 16)
        )

        self.up_btn = tk.Button(
            frame,
            text="上",
            width=6,
            height=2,
            font=btn_font,
            bg="#4ad97a",
            fg="white",
            activebackground="#35bd5e",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
        )
        self.up_btn.grid(row=1, column=1, pady=(0, 8))

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
        self.left_btn.grid(row=2, column=0, padx=(0, 12))

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
        self.right_btn.grid(row=2, column=2, padx=(12, 0))

        self.down_btn = tk.Button(
            frame,
            text="下",
            width=6,
            height=2,
            font=btn_font,
            bg="#d9a44a",
            fg="white",
            activebackground="#bd8a35",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
        )
        self.down_btn.grid(row=3, column=1, pady=(8, 0))

        self.status_var = tk.StringVar(value="状态: 停止")
        tk.Label(frame, textvariable=self.status_var, font=status_font, fg="#444").grid(
            row=4, column=0, columnspan=3, pady=(16, 0)
        )

        tk.Label(
            frame,
            text=f"CoAP: {self.host}:{COAP_PORT}/servo  |  左右=GPIO40  上下=GPIO41",
            font=tkfont.Font(size=10),
            fg="#888",
        ).grid(row=5, column=0, columnspan=3, pady=(8, 0))

        for btn, side in (
            (self.up_btn, "up"),
            (self.down_btn, "down"),
            (self.left_btn, "left"),
            (self.right_btn, "right"),
        ):
            btn.bind("<ButtonPress-1>", lambda e, s=side: self.on_press(s))
            btn.bind("<ButtonRelease-1>", lambda e, s=side: self.on_release(s))
            btn.bind("<Leave>", lambda e, s=side: self.on_leave(s))

        root.bind("<ButtonRelease-1>", self.on_global_release)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _servo_for(self, side: str) -> int:
        return SERVO_V if side in ("up", "down") else SERVO_H

    def _status_for(self, side: str) -> str:
        labels = {
            "left": "状态: 左舵机逆时针",
            "right": "状态: 右舵机顺时针",
            "up": "状态: 上舵机逆时针",
            "down": "状态: 下舵机顺时针",
        }
        return labels[side]

    def on_press(self, side: str) -> None:
        self._pressed.clear()
        self._pressed.add(side)
        if side == "left":
            self.client.left()
        elif side == "right":
            self.client.right()
        elif side == "up":
            self.client.up()
        else:
            self.client.down()
        self.set_status(self._status_for(side))

    def on_release(self, side: str) -> None:
        if side in self._pressed:
            self._pressed.discard(side)
            self.client.stop(self._servo_for(side))
            if not self._pressed:
                self.set_status("状态: 停止")

    def on_leave(self, side: str) -> None:
        self.on_release(side)

    def on_global_release(self, _event=None) -> None:
        if self._pressed:
            for side in list(self._pressed):
                self.client.stop(self._servo_for(side))
            self._pressed.clear()
            self.set_status("状态: 停止")

    def on_close(self) -> None:
        self.client.stop_all()
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
    print(f"舵机控制: coap://{host}:{COAP_PORT}/servo (servo:0=GPIO40左右, servo:1=GPIO41上下)")

    threading.Thread(target=imu_poll_loop, args=(host,), daemon=True).start()
    threading.Thread(target=stream_loop, args=(host,), daemon=True).start()

    root = tk.Tk()
    client = ServoClient(host=host)
    ServoControlApp(root, client, host=host)
    root.mainloop()


if __name__ == "__main__":
    main()

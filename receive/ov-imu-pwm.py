#!/usr/bin/env python3
"""CoAP 客户端：/stream 视频 + /imu 1Hz + SG90 舵机 GUI 控制"""

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
RECONNECT_DELAY = 0.5
RECONNECT_BACKOFF_MAX = 5.0
SERVO_TIMEOUT = 1.0
IMU_INTERVAL = 1.0

STREAM_TOKEN = b"\xca\xfe\xba\xbe"
IMU_TOKEN = b"\xde\xad\xbe\xef"
SERVO_TOKEN = b"\xbe\xef\xca\xfe"

# SG90 角度定位：0~180°，按住按钮步进扫角，松开保持当前位置
SERVO_ANGLE_MIN = 0
SERVO_ANGLE_MAX = 180
SERVO_ANGLE_CENTER = 90
SERVO_STEP_DEG = 3
SERVO_TICK_MS = 40

SERVO_H = 0  # GPIO40 左右
SERVO_V = 1  # GPIO41 上下

ROOT_DIR = Path(__file__).resolve().parent
FRAME_PATH = ROOT_DIR / "frame.jpg"
FRAME_TMP_PATH = ROOT_DIR / "frame.tmp"


def make_coap_socket(host: str, rcvbuf: int = 1024 * 1024) -> socket.socket:
    """创建已 connect 的 CoAP UDP 套接字（Linux 上有利于 conntrack 放行回包）。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    except OSError:
        pass
    sock.connect((host, COAP_PORT))
    return sock


def probe_inbound(host: str, timeout: float = 2.0) -> tuple[bool, str]:
    """探测 ESP32 回包能否到达本机（舵机只发不收，此项可区分入站 UDP 问题）。"""
    try:
        sock = make_coap_socket(host, rcvbuf=64 * 1024)
        sock.settimeout(timeout)
        sock.send(build_get("imu", 1, IMU_TOKEN))
        data = sock.recv(1024)
        sock.close()
        _, _, payload = parse_packet(data)
        if not payload:
            return False, "收到 CoAP 包但无 payload"
        json.loads(payload.decode())
        return True, "IMU 回包正常"
    except socket.timeout:
        return False, f"{timeout:.0f}s 内未收到 /imu 回包（入站 UDP 可能被防火墙/路由丢弃）"
    except OSError as e:
        return False, f"网络错误: {e}"
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
        return False, f"回包解析失败: {e}"


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
    logged_timeout = False
    backoff = RECONNECT_DELAY

    while True:
        sock = None
        try:
            sock = make_coap_socket(host)
            actual_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            if actual_buf < 512 * 1024:
                print(
                    f"[stream] 警告: UDP 接收缓冲仅 {actual_buf} 字节，"
                    f"Linux 可执行 sudo sysctl -w net.core.rmem_max=8388608",
                    file=sys.stderr,
                )
            sock.settimeout(PACKET_TIMEOUT)
            assembler = JpegStreamAssembler(STREAM_TOKEN)
            last_packet = time.monotonic()
            last_keepalive = 0.0
            logged_timeout = False
            backoff = RECONNECT_DELAY

            print(f"[stream] 已连接，订阅 /stream", file=sys.stderr)
            sock.send(build_get("stream", msg_id, STREAM_TOKEN))
            msg_id = (msg_id + 1) & 0xFFFF
            last_packet = time.monotonic()

            while True:
                try:
                    data = sock.recv(65536)
                    last_packet = time.monotonic()
                    logged_timeout = False
                except socket.timeout:
                    now = time.monotonic()
                    idle = now - last_packet
                    if idle >= IDLE_RESUBSCRIBE:
                        if not logged_timeout:
                            print(
                                f"[stream] {idle:.0f}s 无数据，重新订阅 /stream",
                                file=sys.stderr,
                            )
                        break
                    if idle >= KEEPALIVE_INTERVAL and now - last_keepalive >= KEEPALIVE_INTERVAL:
                        if not logged_timeout:
                            print(
                                f"[stream] {idle:.0f}s 无数据，发送 keepalive",
                                file=sys.stderr,
                            )
                            logged_timeout = True
                        sock.send(build_get("stream", msg_id, STREAM_TOKEN))
                        msg_id = (msg_id + 1) & 0xFFFF
                        last_keepalive = now
                    continue

                jpg = assembler.feed(data)
                if jpg:
                    FRAME_TMP_PATH.write_bytes(jpg)
                    os.replace(FRAME_TMP_PATH, FRAME_PATH)
        except OSError as e:
            print(f"[stream] 连接断开: {e}，{backoff:.1f}s 后重连", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 1.5, RECONNECT_BACKOFF_MAX)
        finally:
            if sock is not None:
                sock.close()

        time.sleep(0.3)
        msg_id = (msg_id + 1) & 0xFFFF


def imu_poll_loop(host: str) -> None:
    backoff = RECONNECT_DELAY
    while True:
        sock = None
        try:
            sock = make_coap_socket(host, rcvbuf=64 * 1024)
            sock.settimeout(3.0)
            msg_id = 1
            backoff = RECONNECT_DELAY
            while True:
                sock.send(build_get("imu", msg_id, IMU_TOKEN))
                data = sock.recv(1024)
                _, _, payload = parse_packet(data)
                if payload:
                    imu = json.loads(payload.decode())
                    print(
                        f"加速度: {imu['ax']:.4f} {imu['ay']:.4f} {imu['az']:.4f} | "
                        f"陀螺仪: {imu['gx']:.4f} {imu['gy']:.4f} {imu['gz']:.4f}"
                    )
                msg_id = (msg_id + 1) & 0xFFFF
                time.sleep(IMU_INTERVAL)
        except socket.timeout:
            print(f"[imu] 超时未收到回包，{backoff:.1f}s 后重连", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 1.5, RECONNECT_BACKOFF_MAX)
        except OSError as e:
            print(f"[imu] 连接断开: {e}，{backoff:.1f}s 后重连", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 1.5, RECONNECT_BACKOFF_MAX)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[imu] 数据解析失败: {e}", file=sys.stderr)
            time.sleep(1.0)
        finally:
            if sock is not None:
                sock.close()


class ServoClient:
    def __init__(self, host: str = COAP_HOST, port: int = COAP_PORT):
        self.host = host
        self.port = port
        self._msg_id = 100
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(SERVO_TIMEOUT)
        self._angle = {SERVO_H: SERVO_ANGLE_CENTER, SERVO_V: SERVO_ANGLE_CENTER}

    def close(self) -> None:
        self._sock.close()

    def angle(self, servo: int = SERVO_H) -> int:
        return self._angle.get(servo, SERVO_ANGLE_CENTER)

    def send_angle(self, angle: int, servo: int = SERVO_H) -> None:
        angle = max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, int(angle)))
        with self._lock:
            # 角度未变则不发，避免重复刷新 PWM 导致抖动
            if self._angle.get(servo) == angle:
                return
            self._angle[servo] = angle
            payload = json.dumps({"servo": servo, "angle": angle}).encode()
            req = build_put("servo", self._msg_id, SERVO_TOKEN, payload)
            self._msg_id = (self._msg_id + 1) & 0xFFFF
        try:
            self._sock.sendto(req, (self.host, self.port))
        except OSError as e:
            print(f"舵机发送失败: {e}", file=sys.stderr)

    def nudge(self, servo: int, delta: int) -> int:
        """相对步进，返回新角度。"""
        new_angle = self.angle(servo) + delta
        self.send_angle(new_angle, servo)
        return self.angle(servo)


class ServoControlApp:
    # 按住时的步进方向：左右=水平舵机，上下=垂直舵机
    _DELTAS = {
        "left": (SERVO_H, -SERVO_STEP_DEG),
        "right": (SERVO_H, SERVO_STEP_DEG),
        "up": (SERVO_V, -SERVO_STEP_DEG),
        "down": (SERVO_V, SERVO_STEP_DEG),
    }

    def __init__(self, root: tk.Tk, client: ServoClient, host: str = COAP_HOST):
        self.root = root
        self.client = client
        self.host = host
        self._pressed: set[str] = set()
        self._tick_id: Optional[str] = None

        root.title("OV3660 + IMU + SG90 双舵机")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        title_font = tkfont.Font(size=16, weight="bold")
        btn_font = tkfont.Font(size=28, weight="bold")
        status_font = tkfont.Font(size=12)

        frame = tk.Frame(root, padx=24, pady=20)
        frame.pack()

        tk.Label(frame, text="按住转动，松开保持", font=title_font).grid(
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

        self.status_var = tk.StringVar(value="状态: 保持")
        tk.Label(frame, textvariable=self.status_var, font=status_font, fg="#444").grid(
            row=4, column=0, columnspan=3, pady=(16, 0)
        )

        tk.Label(
            frame,
            text=f"SG90  |  CoAP: {self.host}:{COAP_PORT}/servo  |  左右=GPIO40  上下=GPIO41",
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

    def _hold_status(self) -> str:
        return "状态: 保持"

    def _move_status(self, side: str) -> str:
        labels = {
            "left": "左转",
            "right": "右转",
            "up": "上仰",
            "down": "下俯",
        }
        return f"状态: {labels[side]}"

    def _cancel_tick(self) -> None:
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None

    def _tick(self) -> None:
        self._tick_id = None
        if not self._pressed:
            return
        side = next(iter(self._pressed))
        servo, delta = self._DELTAS[side]
        self.client.nudge(servo, delta)
        self.set_status(self._move_status(side))
        self._tick_id = self.root.after(SERVO_TICK_MS, self._tick)

    def on_press(self, side: str) -> None:
        self._cancel_tick()
        self._pressed.clear()
        self._pressed.add(side)
        servo, delta = self._DELTAS[side]
        self.client.nudge(servo, delta)
        self.set_status(self._move_status(side))
        self._tick_id = self.root.after(SERVO_TICK_MS, self._tick)

    def on_release(self, side: str) -> None:
        if side in self._pressed:
            self._pressed.discard(side)
            self._cancel_tick()
            # SG90 松开后保持当前位置，不再回中
            self.set_status(self._hold_status())

    def on_leave(self, side: str) -> None:
        self.on_release(side)

    def on_global_release(self, _event=None) -> None:
        if self._pressed:
            self._pressed.clear()
            self._cancel_tick()
            self.set_status(self._hold_status())

    def on_close(self) -> None:
        self._cancel_tick()
        self.client.close()
        self.root.destroy()


def _parse_main_args(argv: list[str]) -> tuple[str, bool]:
    host = COAP_HOST
    headless = False
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print("用法: ov-imu-pwm.py [选项] [ESP32_IP]")
            print("  无参数: 打开舵机控制窗口，同时订阅视频流与 IMU")
            print("  带 IP:  指定 ESP32 热点 IP (默认 192.168.4.1)")
            print()
            print("选项:")
            print("  --headless   无 GUI，仅后台拉流写 frame.jpg 并打印 IMU（供 test-web/server 调用）")
            print()
            print(f"  视频帧保存至 {FRAME_PATH}")
            print(f"  IMU 数据以 {1/IMU_INTERVAL:.0f}Hz 打印到终端")
            sys.exit(0)
        elif arg == "--headless":
            headless = True
        elif not arg.startswith("-"):
            host = arg
        i += 1
    return host, headless


def _start_stream_services(host: str) -> None:
    print(f"视频流: coap://{host}:{COAP_PORT}/stream")
    print(f"IMU 通道: coap://{host}:{COAP_PORT}/imu ({1/IMU_INTERVAL:.0f}Hz)")
    print(
        f"舵机控制: coap://{host}:{COAP_PORT}/servo "
        f"(SG90 角度0~180, servo:0=GPIO40左右, servo:1=GPIO41上下)"
    )

    ok, detail = probe_inbound(host)
    if ok:
        print(f"入站探测: {detail}")
    else:
        print(f"入站探测失败: {detail}", file=sys.stderr)
        print(
            "舵机仍可控制（仅出站 UDP），但 IMU/视频需要 ESP32 回包。\n"
            "Linux 排查: ping 192.168.4.1 | iw dev wlan0 set power_save off | "
            "sudo ufw disable（测试）| sysctl net.core.rmem_max",
            file=sys.stderr,
        )

    threading.Thread(target=imu_poll_loop, args=(host,), daemon=True).start()
    threading.Thread(target=stream_loop, args=(host,), daemon=True).start()


def run_headless(host: str) -> None:
    _start_stream_services(host)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[headless] 已退出")


def main() -> None:
    host, headless = _parse_main_args(sys.argv)
    if headless:
        run_headless(host)
        return

    _start_stream_services(host)

    root = tk.Tk()
    client = ServoClient(host=host)
    ServoControlApp(root, client, host=host)
    root.mainloop()


if __name__ == "__main__":
    main()

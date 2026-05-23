import io
import os
import socket
import struct
import time

from PIL import Image

UDP_PORT = 1234
ESP_IP = "192.168.4.1"
SYNC = 0xAA55AA55
HDR = 20
MAX_PKT = 1400  # 必须与 ov5640-udp.ino 中 MAX_PKT_DATA 一致
MAX_FRAME = 256 * 1024
FRAME_TIMEOUT = 3.0
HELLO_INTERVAL = 2.0
STATUS_INTERVAL = 2.0

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "receive.jpg")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
sock.settimeout(0.001)
sock.bind(("0.0.0.0", UDP_PORT))


def send_hello():
    sock.sendto(b"hello", (ESP_IP, UDP_PORT))


def parse_packet(pkt):
    if len(pkt) < HDR + 1:
        return None
    sync, fid, offset, data_len, total_len, pw, ph = struct.unpack("<IHIHIHH", pkt[:HDR])
    if sync != SYNC:
        return None
    if data_len < 1 or data_len > MAX_PKT:
        return None
    if len(pkt) < HDR + data_len:
        return None
    if total_len < 200 or total_len > MAX_FRAME:
        return None
    if offset + data_len > total_len:
        return None
    return {
        "fid": fid,
        "offset": offset,
        "data_len": data_len,
        "total": total_len,
        "payload": pkt[HDR : HDR + data_len],
    }


def frame_done(st):
    return len(st["chunks"]) >= st["n_chunks"]


send_hello()
print(f"保存: {SAVE_PATH}")
print("已连 WiFi ESP32S3-OV5640，收包中…")

frames = {}
latest_fid = -1
stats = {"pkt": 0, "ok": 0, "save": 0, "bad_sync": 0, "timeout": 0}
last_status = time.time()
last_hello = time.time()
last_print = 0.0
fps_count = 0
fps_t0 = time.time()


def handle_packet(pkt, now):
    global latest_fid

    stats["pkt"] += 1
    if len(pkt) >= 4 and struct.unpack_from("<I", pkt, 0)[0] != SYNC:
        stats["bad_sync"] += 1

    info = parse_packet(pkt)
    if not info:
        return

    stats["ok"] += 1
    fid = info["fid"]
    offset = info["offset"]
    total_len = info["total"]
    payload = info["payload"]

    if fid > latest_fid:
        latest_fid = fid
        for old in list(frames.keys()):
            if old < fid - 2:
                del frames[old]

    if fid < latest_fid - 1:
        return

    if fid not in frames:
        n_chunks = (total_len + MAX_PKT - 1) // MAX_PKT
        frames[fid] = {
            "total": total_len,
            "n_chunks": n_chunks,
            "buf": bytearray(total_len),
            "chunks": set(),
            "ts": now,
        }

    st = frames[fid]
    if total_len != st["total"]:
        return

    ci = offset // MAX_PKT
    if ci in st["chunks"]:
        return

    st["buf"][offset : offset + len(payload)] = payload
    st["chunks"].add(ci)
    st["ts"] = now

    if not frame_done(st):
        return

    data = bytes(st["buf"][: st["total"]])
    del frames[fid]

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return

    img.save(SAVE_PATH, quality=90)
    stats["save"] += 1

    global fps_count, fps_t0, last_print
    fps_count += 1
    if now - last_print >= 1.0:
        dt = now - fps_t0
        print(
            f"✓ {img.size[0]}x{img.size[1]} ~{fps_count / max(dt, 0.001):.1f}fps "
            f"{len(data) // 1024}KB | ok率 {100 * stats['ok'] / max(stats['pkt'], 1):.0f}%"
        )
        fps_count = 0
        fps_t0 = now
        last_print = now


try:
    while True:
        now = time.time()
        if now - last_hello >= HELLO_INTERVAL:
            send_hello()
            last_hello = now

        drained = 0
        while drained < 800:
            try:
                pkt, _ = sock.recvfrom(2048)
                drained += 1
                handle_packet(pkt, now)
            except socket.timeout:
                break

        for fid in list(frames.keys()):
            if now - frames[fid]["ts"] > FRAME_TIMEOUT:
                stats["timeout"] += 1
                del frames[fid]

        if now - last_status >= STATUS_INTERVAL:
            last_status = now
            prog = "".join(
                f" #{k}:{len(v['chunks'])}/{v['n_chunks']}" for k, v in frames.items()
            )
            rate = 100 * stats["ok"] / max(stats["pkt"], 1)
            print(
                f"[状态] 包{stats['pkt']} 有效{stats['ok']}({rate:.0f}%) "
                f"存{stats['save']} 非同步{stats['bad_sync']} 超时{stats['timeout']}"
                f"{prog or ' 拼帧中…'}"
            )

except KeyboardInterrupt:
    print(f"\n退出，共保存 {stats['save']} 张 -> {SAVE_PATH}")
finally:
    sock.close()

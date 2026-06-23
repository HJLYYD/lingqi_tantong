#pragma once
#include <WiFiUdp.h>
#include <stdint.h>
#include <string.h>

#define COAP_PORT           5683
#define COAP_MAX_PACKET     1400
#define COAP_DEFAULT_SZX    6

#define STREAM_FRAME_GAP_MS     28
#define STREAM_BLOCK_GAP_MS     1
#define STREAM_BLOCKS_PER_TICK  6

#define COAP_TYPE_CON       0
#define COAP_TYPE_NON       1
#define COAP_TYPE_ACK       2

#define COAP_CODE_GET       0x01
#define COAP_CODE_PUT       0x03
#define COAP_CODE_CONTENT   0x45
#define COAP_CODE_CHANGED   0x44
#define COAP_CODE_BAD       0x80
#define COAP_CODE_NOT_FOUND 0x84

#define COAP_OPT_ETAG           4
#define COAP_OPT_URI_PATH      11
#define COAP_OPT_CONTENT_FMT   12
#define COAP_OPT_BLOCK2        23
#define COAP_OPT_SIZE2         28

#define COAP_FMT_JSON       50
#define COAP_FMT_OCTET      42
#define COAP_FMT_JPEG       22

typedef struct {
  uint8_t  type;
  uint8_t  code;
  uint16_t msg_id;
  uint8_t  token[8];
  uint8_t  token_len;
  char     uri[32];
  bool     has_block2;
  uint32_t block2_num;
  uint8_t  block2_szx;
} CoapRequest;

typedef void (*CoapFrameProvider)(uint8_t** out, size_t* out_len, bool refresh, bool with_imu);
typedef void (*CoapImuProvider)(char* buf, size_t buf_size, size_t* out_len);
typedef bool (*CoapServoHandler)(int servo_id, int angle, char* resp, size_t resp_size, size_t* resp_len);

class CoapServer {
public:
  void begin() {
    _udp.begin(COAP_PORT);
  }

  void setFrameProvider(CoapFrameProvider fn) { _frame_fn = fn; }
  void setImuProvider(CoapImuProvider fn)   { _imu_fn = fn; }
  void setServoHandler(CoapServoHandler fn) { _servo_fn = fn; }

  bool isBusy() const { return _stream.busy; }
  bool isStreamActive() const { return _stream.active; }

  void loop() {
    int pkt_len = _udp.parsePacket();
    if (pkt_len <= 0) return;

    IPAddress remote = _udp.remoteIP();
    uint16_t remote_port = _udp.remotePort();

    uint8_t rx[COAP_MAX_PACKET];
    if (pkt_len > (int)sizeof(rx)) pkt_len = sizeof(rx);
    pkt_len = _udp.read(rx, pkt_len);

    CoapRequest req;
    const uint8_t* payload = NULL;
    size_t payload_len = 0;
    if (!_parseRequest(rx, pkt_len, &req, &payload, &payload_len)) {
      _sendResponse(remote, remote_port, COAP_TYPE_ACK, COAP_CODE_BAD,
                    req.token, req.token_len, req.msg_id,
                    NULL, 0, COAP_FMT_OCTET, false, 0, false, 0, 0, 0);
      return;
    }

    if (req.code == COAP_CODE_PUT && strcmp(req.uri, "servo") == 0) {
      _handleServoPut(remote, remote_port, &req, payload, payload_len);
      return;
    }

    if (req.code != COAP_CODE_GET) {
      _sendResponse(remote, remote_port, COAP_TYPE_ACK, COAP_CODE_BAD,
                    req.token, req.token_len, req.msg_id,
                    NULL, 0, COAP_FMT_OCTET, false, 0, false, 0, 0, 0);
      return;
    }

    if (strcmp(req.uri, "imu") == 0) {
      _handleImu(remote, remote_port, &req);
    } else if (strcmp(req.uri, "stream") == 0) {
      _registerStream(remote, remote_port, &req);
      _stream.tx.in_progress = false;
      _stream.tx.last_frame_ms = 0;
    } else if (strcmp(req.uri, "frame") == 0 || strcmp(req.uri, "") == 0) {
      _handleFrame(remote, remote_port, &req);
    } else {
      _sendResponse(remote, remote_port, COAP_TYPE_ACK, COAP_CODE_NOT_FOUND,
                    req.token, req.token_len, req.msg_id,
                    NULL, 0, COAP_FMT_OCTET, false, 0, false, 0, 0, 0);
    }
  }

  // 稳定分块推送：发完一整帧再抓下一帧，固定 burst 避免 UDP 丢包导致断流
  void tickStream() {
    if (!_stream.active || !_frame_fn || _stream.busy) return;

    unsigned long now = millis();

    if (_stream.tx.in_progress) {
      if (now - _stream.tx.last_block_ms < STREAM_BLOCK_GAP_MS) return;

      _stream.busy = true;

      const uint8_t* frame_data = _stream.tx.frame_data;
      size_t frame_len = _stream.tx.frame_len;
      if (!frame_data || frame_len == 0) {
        _stream.tx.in_progress = false;
        _stream.busy = false;
        return;
      }

      size_t block_size = _blockSizeFromSzx(_stream.tx.block_szx);
      uint32_t total_blocks = (frame_len + block_size - 1) / block_size;

      int sent = 0;
      while (sent < STREAM_BLOCKS_PER_TICK && _stream.tx.next_block < total_blocks) {
        _sendBlock(_stream.ip, _stream.port, COAP_TYPE_NON,
                   (_stream.msg_id + _stream.tx.next_block) & 0xFFFF,
                   _stream.token, _stream.token_len,
                   frame_data, frame_len,
                   _stream.tx.next_block, _stream.tx.block_szx, _stream.tx.etag,
                   COAP_FMT_JPEG);
        _stream.tx.next_block++;
        sent++;
      }
      _stream.tx.last_block_ms = now;

      if (_stream.tx.next_block >= total_blocks) {
        _stream.tx.in_progress = false;
        _stream.tx.last_frame_ms = now;
        _stream.msg_id = (_stream.msg_id + total_blocks) & 0xFFFF;
      }

      _stream.busy = false;
      return;
    }

    if (now - _stream.tx.last_frame_ms < STREAM_FRAME_GAP_MS) return;

    _stream.busy = true;

    uint8_t* frame_data = nullptr;
    size_t frame_len = 0;
    _frame_fn(&frame_data, &frame_len, true, false);
    if (!frame_data || frame_len == 0) {
      _stream.busy = false;
      return;
    }

    _frame_etag++;
    _stream.tx.frame_data = frame_data;
    _stream.tx.frame_len = frame_len;
    _stream.tx.next_block = 0;
    _stream.tx.etag = _frame_etag;
    _stream.tx.block_szx = COAP_DEFAULT_SZX;
    _stream.tx.in_progress = true;
    _stream.tx.last_block_ms = 0;

    size_t block_size = _blockSizeFromSzx(_stream.tx.block_szx);
    uint32_t total_blocks = (frame_len + block_size - 1) / block_size;

    int sent = 0;
    while (sent < STREAM_BLOCKS_PER_TICK && _stream.tx.next_block < total_blocks) {
      _sendBlock(_stream.ip, _stream.port, COAP_TYPE_NON,
                 (_stream.msg_id + _stream.tx.next_block) & 0xFFFF,
                 _stream.token, _stream.token_len,
                 frame_data, frame_len,
                 _stream.tx.next_block, _stream.tx.block_szx, _stream.tx.etag,
                 COAP_FMT_JPEG);
      _stream.tx.next_block++;
      sent++;
    }
    _stream.tx.last_block_ms = now;

    if (_stream.tx.next_block >= total_blocks) {
      _stream.tx.in_progress = false;
      _stream.tx.last_frame_ms = now;
      _stream.msg_id = (_stream.msg_id + total_blocks) & 0xFFFF;
    }

    _stream.busy = false;
  }

private:
  WiFiUDP _udp;
  CoapFrameProvider _frame_fn = nullptr;
  CoapImuProvider   _imu_fn   = nullptr;
  CoapServoHandler  _servo_fn = nullptr;
  uint32_t _frame_etag = 0;

  struct {
    bool active;
    bool busy;
    IPAddress ip;
    uint16_t port;
    uint8_t token[8];
    uint8_t token_len;
    uint16_t msg_id;
    unsigned long last_push;
    struct {
      bool in_progress;
      size_t frame_len;
      uint32_t next_block;
      uint32_t etag;
      uint8_t block_szx;
      unsigned long last_block_ms;
      unsigned long last_frame_ms;
      const uint8_t* frame_data;
    } tx;
  } _stream = {false, false, {}, 0, {}, 0, 0, 0,
               {false, 0, 0, 0, COAP_DEFAULT_SZX, 0, 0, nullptr}};

  static uint16_t _readUint16(const uint8_t* p) {
    return ((uint16_t)p[0] << 8) | p[1];
  }

  static void _writeUint16(uint8_t* p, uint16_t v) {
    p[0] = (v >> 8) & 0xFF;
    p[1] = v & 0xFF;
  }

  static size_t _blockSizeFromSzx(uint8_t szx) {
    if (szx > 6) szx = 6;
    return (size_t)16 << szx;
  }

  static int _encodeUintOption(uint8_t* out, int prev_opt, int opt_num, uint32_t val) {
    int delta = opt_num - prev_opt;
    uint8_t val_bytes[4];
    int val_len = 0;

    if (val <= 255) {
      val_bytes[0] = (uint8_t)val;
      val_len = 1;
    } else if (val <= 65535) {
      val_bytes[0] = (val >> 8) & 0xFF;
      val_bytes[1] = val & 0xFF;
      val_len = 2;
    } else {
      val_bytes[0] = (val >> 16) & 0xFF;
      val_bytes[1] = (val >> 8) & 0xFF;
      val_bytes[2] = val & 0xFF;
      val_len = 3;
    }

    int len = 0;
    if (delta <= 12 && val_len <= 12) {
      out[0] = (delta << 4) | val_len;
      len = 1;
    } else if (delta <= 12) {
      out[0] = (delta << 4) | 13;
      out[1] = val_len - 13;
      len = 2;
    } else if (delta <= 255 && val_len <= 12) {
      out[0] = 13 << 4 | val_len;
      out[1] = delta - 13;
      len = 2;
    } else {
      return 0;
    }

    memcpy(out + len, val_bytes, val_len);
    return len + val_len;
  }

  static int _encodeBlock2(uint8_t* out, int prev_opt, uint32_t num, bool more, uint8_t szx) {
    uint32_t val = (num << 4) | (more ? 0x08u : 0x00u) | (szx & 0x07u);
    return _encodeUintOption(out, prev_opt, COAP_OPT_BLOCK2, val);
  }

  bool _parseRequest(const uint8_t* data, int len, CoapRequest* req,
                     const uint8_t** payload_out, size_t* payload_len_out) {
    if (len < 4) return false;

    if (payload_out) *payload_out = NULL;
    if (payload_len_out) *payload_len_out = 0;

    memset(req, 0, sizeof(CoapRequest));
    req->type = (data[0] >> 4) & 0x03;
    req->code = data[1];
    req->msg_id = _readUint16(data + 2);
    req->token_len = data[0] & 0x0F;
    if ((data[0] >> 6) != 1) return false;
    if (req->token_len > 8) return false;

    int pos = 4;
    for (int i = 0; i < req->token_len; i++) {
      if (pos >= len) return false;
      req->token[i] = data[pos++];
    }

    int opt = 0;
    bool has_payload = false;
    while (pos < len) {
      if (data[pos] == 0xFF) {
        pos++;
        has_payload = true;
        break;
      }

      uint8_t hdr = data[pos++];
      int delta = (hdr >> 4) & 0x0F;
      int olen  = hdr & 0x0F;
      if (delta == 13) { if (pos >= len) return false; delta = data[pos++] + 13; }
      else if (delta == 14) { if (pos + 1 >= len) return false; delta = data[pos++] << 8; delta |= data[pos++]; delta += 269; }
      else if (delta == 15) return false;

      if (olen == 13) { if (pos >= len) return false; olen = data[pos++] + 13; }
      else if (olen == 14) { if (pos + 1 >= len) return false; olen = data[pos++] << 8; olen |= data[pos++]; olen += 269; }
      else if (olen == 15) return false;

      opt += delta;
      if (pos + olen > len) return false;

      if (opt == COAP_OPT_URI_PATH) {
        if (req->uri[0]) strlcat(req->uri, "/", sizeof(req->uri));
        size_t cur = strlen(req->uri);
        size_t copy = olen;
        if (cur + copy >= sizeof(req->uri)) copy = sizeof(req->uri) - cur - 1;
        memcpy(req->uri + cur, data + pos, copy);
        req->uri[cur + copy] = '\0';
      } else if (opt == COAP_OPT_BLOCK2 && olen >= 1) {
        req->has_block2 = true;
        uint32_t val = 0;
        for (int i = 0; i < olen; i++) val = (val << 8) | data[pos + i];
        req->block2_num = val >> 4;
        req->block2_szx = val & 0x07;
      }

      pos += olen;
    }

    if (has_payload) {
      if (payload_out) *payload_out = data + pos;
      if (payload_len_out) *payload_len_out = (size_t)(len - pos);
    }

    return true;
  }

  static bool _parseServoFromPayload(const uint8_t* payload, size_t payload_len,
                                     int* servo_id_out, int* angle_out) {
    if (!payload || payload_len == 0 || !servo_id_out || !angle_out) return false;

    char buf[64];
    size_t copy = payload_len;
    if (copy >= sizeof(buf)) copy = sizeof(buf) - 1;
    memcpy(buf, payload, copy);
    buf[copy] = '\0';

    while (copy > 0 && (buf[copy - 1] == ' ' || buf[copy - 1] == '\r' || buf[copy - 1] == '\n')) {
      buf[--copy] = '\0';
    }

    *servo_id_out = 0;
    *angle_out = -1;

    const char* servo_key = strstr(buf, "\"servo\"");
    if (servo_key) {
      const char* p = strchr(servo_key + 7, ':');
      if (!p) return false;
      p++;
      while (*p == ' ' || *p == '\t') p++;
      int servo_id = atoi(p);
      if (servo_id < 0 || servo_id > 1) return false;
      *servo_id_out = servo_id;
    }

    const char* angle_key = strstr(buf, "\"angle\"");
    if (!angle_key) return false;
    const char* p = strchr(angle_key + 7, ':');
    if (!p) return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    int angle = atoi(p);
    if (angle < 0 || angle > 180) return false;
    *angle_out = angle;
    return true;
  }

  int _buildResponse(uint8_t* out, int out_max,
                     uint8_t type, uint8_t code, uint16_t msg_id,
                     const uint8_t* token, uint8_t token_len,
                     uint32_t etag, uint8_t content_format,
                     bool has_block2, uint32_t block_num, bool block_more, uint8_t block_szx,
                     size_t total_size,
                     const uint8_t* payload, size_t payload_len) {
    if (out_max < 16) return 0;

    out[0] = (1 << 6) | ((type & 0x03) << 4) | (token_len & 0x0F);
    out[1] = code;
    _writeUint16(out + 2, msg_id);

    int pos = 4;
    for (int i = 0; i < token_len; i++) out[pos++] = token[i];

    int prev_opt = 0;
    int n;

    if (etag != 0) {
      n = _encodeUintOption(out + pos, prev_opt, COAP_OPT_ETAG, etag);
      if (n <= 0) return 0;
      pos += n;
      prev_opt = COAP_OPT_ETAG;
    }

    n = _encodeUintOption(out + pos, prev_opt, COAP_OPT_CONTENT_FMT, content_format);
    if (n <= 0) return 0;
    pos += n;
    prev_opt = COAP_OPT_CONTENT_FMT;

    if (has_block2) {
      n = _encodeBlock2(out + pos, prev_opt, block_num, block_more, block_szx);
      if (n <= 0) return 0;
      pos += n;
      prev_opt = COAP_OPT_BLOCK2;
    }

    if (has_block2 && total_size > 0) {
      n = _encodeUintOption(out + pos, prev_opt, COAP_OPT_SIZE2, (uint32_t)total_size);
      if (n <= 0) return 0;
      pos += n;
    }

    if (payload_len > 0) {
      if (pos + 1 + (int)payload_len > out_max) {
        payload_len = out_max - pos - 1;
      }
      out[pos++] = 0xFF;
      memcpy(out + pos, payload, payload_len);
      pos += payload_len;
    }

    return pos;
  }

  void _sendBlock(IPAddress ip, uint16_t port,
                  uint8_t type, uint16_t msg_id,
                  const uint8_t* token, uint8_t token_len,
                  const uint8_t* frame_data, size_t frame_len,
                  uint32_t block_num, uint8_t block_szx, uint32_t etag,
                  uint8_t content_format) {
    size_t block_size = _blockSizeFromSzx(block_szx);
    size_t offset = block_num * block_size;
    if (offset >= frame_len) return;

    size_t remaining = frame_len - offset;
    size_t send_len = (remaining > block_size) ? block_size : remaining;
    bool more = (offset + send_len) < frame_len;

    uint8_t tx[COAP_MAX_PACKET];
    int len = _buildResponse(tx, sizeof(tx), type, COAP_CODE_CONTENT, msg_id,
                             token, token_len, etag, content_format,
                             true, block_num, more, block_szx, frame_len,
                             frame_data + offset, send_len);
    if (len <= 0) return;

    _udp.beginPacket(ip, port);
    _udp.write(tx, len);
    _udp.endPacket();
  }

  void _burstSendFrame(IPAddress ip, uint16_t port,
                       const uint8_t* token, uint8_t token_len, uint16_t msg_id,
                       uint8_t rsp_type, uint8_t* frame_data, size_t frame_len,
                       uint8_t block_szx, uint32_t etag, uint8_t content_format) {
    size_t block_size = _blockSizeFromSzx(block_szx);
    uint32_t total_blocks = (frame_len + block_size - 1) / block_size;

    for (uint32_t b = 0; b < total_blocks; b++) {
      uint8_t type = COAP_TYPE_NON;
      _sendBlock(ip, port, type, (msg_id + b) & 0xFFFF,
                 token, token_len, frame_data, frame_len, b, block_szx, etag,
                 content_format);
    }
  }

  void _registerStream(IPAddress ip, uint16_t port, CoapRequest* req) {
    _stream.active = true;
    _stream.busy = false;
    _stream.ip = ip;
    _stream.port = port;
    _stream.token_len = req->token_len;
    memcpy(_stream.token, req->token, req->token_len);
    _stream.msg_id = req->msg_id;
    _stream.last_push = millis();
    _stream.tx.in_progress = false;
    _stream.tx.frame_data = nullptr;
    _stream.tx.last_frame_ms = 0;
    _stream.tx.last_block_ms = 0;
  }

  void _pushFrame(IPAddress ip, uint16_t port, CoapRequest* req,
                  bool capture, bool with_imu) {
    uint8_t block_szx = req->has_block2 ? req->block2_szx : COAP_DEFAULT_SZX;

    uint8_t* frame_data = nullptr;
    size_t frame_len = 0;
    if (_frame_fn) _frame_fn(&frame_data, &frame_len, capture, with_imu);

    if (!frame_data || frame_len == 0) {
      uint8_t rsp_type = (req->type == COAP_TYPE_CON) ? COAP_TYPE_ACK : COAP_TYPE_NON;
      _sendResponse(ip, port, rsp_type, COAP_CODE_BAD,
                    req->token, req->token_len, req->msg_id,
                    NULL, 0, COAP_FMT_OCTET, false, 0, false, 0, 0, 0);
      return;
    }

    _frame_etag++;
    uint8_t fmt = with_imu ? COAP_FMT_OCTET : COAP_FMT_JPEG;
    _burstSendFrame(ip, port, req->token, req->token_len, req->msg_id,
                    COAP_TYPE_NON, frame_data, frame_len, block_szx, _frame_etag, fmt);
  }

  void _sendResponse(IPAddress ip, uint16_t port,
                     uint8_t type, uint8_t code,
                     const uint8_t* token, uint8_t token_len, uint16_t msg_id,
                     const uint8_t* payload, size_t payload_len,
                     uint8_t content_format,
                     bool has_block2, uint32_t block_num, bool block_more,
                     uint8_t block_szx, uint32_t etag, size_t total_size) {
    uint8_t tx[COAP_MAX_PACKET];
    size_t block_size = _blockSizeFromSzx(block_szx);
    size_t send_len = payload_len;
    bool more = block_more;
    const uint8_t* send_ptr = payload;

    if (has_block2) {
      size_t offset = block_num * block_size;
      if (offset >= payload_len) {
        send_len = 0;
        more = false;
      } else {
        size_t remaining = payload_len - offset;
        if (remaining > block_size) {
          send_len = block_size;
          more = true;
        } else {
          send_len = remaining;
          more = false;
        }
        send_ptr = payload + offset;
      }
    }

    int len = _buildResponse(tx, sizeof(tx), type, code, msg_id,
                             token, token_len, etag, content_format,
                             has_block2, block_num, more, block_szx, total_size,
                             send_ptr, send_len);
    if (len <= 0) return;

    _udp.beginPacket(ip, port);
    _udp.write(tx, len);
    _udp.endPacket();
  }

  void _handleServoPut(IPAddress ip, uint16_t port, CoapRequest* req,
                       const uint8_t* payload, size_t payload_len) {
    uint8_t rsp_type = (req->type == COAP_TYPE_CON) ? COAP_TYPE_ACK : COAP_TYPE_NON;

    if (!_servo_fn) {
      _sendResponse(ip, port, rsp_type, COAP_CODE_NOT_FOUND,
                    req->token, req->token_len, req->msg_id,
                    NULL, 0, COAP_FMT_OCTET, false, 0, false, 0, 0, 0);
      return;
    }

    int servo_id = 0;
    int angle = -1;
    if (!_parseServoFromPayload(payload, payload_len, &servo_id, &angle)) {
      const char* err = "{\"ok\":false,\"error\":\"servo must be 0-1, angle 0-180\"}";
      _sendResponse(ip, port, rsp_type, COAP_CODE_BAD,
                    req->token, req->token_len, req->msg_id,
                    (const uint8_t*)err, strlen(err), COAP_FMT_JSON,
                    false, 0, false, 0, 0, 0);
      return;
    }

    char json[80];
    size_t json_len = 0;
    if (!_servo_fn(servo_id, angle, json, sizeof(json), &json_len)) {
      const char* err = "{\"ok\":false,\"error\":\"servo failed\"}";
      _sendResponse(ip, port, rsp_type, COAP_CODE_BAD,
                    req->token, req->token_len, req->msg_id,
                    (const uint8_t*)err, strlen(err), COAP_FMT_JSON,
                    false, 0, false, 0, 0, 0);
      return;
    }

    _sendResponse(ip, port, rsp_type, COAP_CODE_CHANGED,
                  req->token, req->token_len, req->msg_id,
                  (const uint8_t*)json, json_len, COAP_FMT_JSON,
                  false, 0, false, 0, 0, 0);
  }

  void _handleImu(IPAddress ip, uint16_t port, CoapRequest* req) {
    char json[128];
    size_t json_len = 0;
    if (_imu_fn) _imu_fn(json, sizeof(json), &json_len);

    uint8_t rsp_type = (req->type == COAP_TYPE_CON) ? COAP_TYPE_ACK : COAP_TYPE_NON;
    _sendResponse(ip, port, rsp_type, COAP_CODE_CONTENT,
                  req->token, req->token_len, req->msg_id,
                  (const uint8_t*)json, json_len, COAP_FMT_JSON,
                  false, 0, false, 0, 0, 0);
  }

  void _handleFrame(IPAddress ip, uint16_t port, CoapRequest* req) {
    uint32_t block_num = req->has_block2 ? req->block2_num : 0;
    if (block_num != 0) return;
    _pushFrame(ip, port, req, true, true);
  }
};

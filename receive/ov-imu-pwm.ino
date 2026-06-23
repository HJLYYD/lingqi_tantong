#include <Wire.h>
#include <math.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_wifi.h"
#include "driver/ledc.h"
#include "coap_server.h"

// Tools > USB Mode: Hardware CDC and JTAG 时，关闭 JTAG 调试桥，仅保留 CDC 串口 (Serial)
#if ARDUINO_USB_MODE
#include "soc/usb_serial_jtag_reg.h"

static void disableUsbJtagKeepCdc() {
  CLEAR_PERI_REG_MASK(USB_SERIAL_JTAG_CONF0_REG, USB_SERIAL_JTAG_USB_JTAG_BRIDGE_EN);
}

__attribute__((constructor(101))) static void earlyDisableUsbJtag() {
  disableUsbJtagKeepCdc();
}
#endif

// ==================== GY85 配置 ====================
#define GY85_SDA 21
#define GY85_SCL 14

#define ADXL345_ADDR 0x53
#define ITG3205_ADDR 0x68

const float ACCEL_SCALE = 2.0 * 9.80665 / 32768.0;
const float GYRO_SCALE = 250.0 * (PI / 180.0) / 32768.0;

int16_t gx_off = 0, gy_off = 0, gz_off = 0;

// ==================== 舵机配置 (LEDC, MG996R 连续旋转) ====================
#define SERVO_PIN 40
#define SERVO_CCW_US 1000   // 逆时针
#define SERVO_STOP_US 1500  // 停止
#define SERVO_CW_US 2000    // 顺时针
#define SERVO_FREQ_HZ 50
#define SERVO_RES_BITS 14
#define SERVO_PERIOD_US 20000UL
#define SERVO_LEDC_TIMER LEDC_TIMER_1
#define SERVO_LEDC_CHANNEL LEDC_CHANNEL_3
#define SERVO_LEDC_MODE LEDC_LOW_SPEED_MODE

enum ServoDir { SERVO_DIR_STOP = 0, SERVO_DIR_CCW = -1, SERVO_DIR_CW = 1 };
ServoDir g_servo_dir = SERVO_DIR_STOP;

// ==================== WiFi 热点配置 ====================
#define AP_SSID "ESP32-Camera-AP"
#define AP_PASSWORD "12345678"

// IMU 采样间隔（毫秒），1Hz
#define IMU_INTERVAL_MS 1000

// ==================== CoAP 帧缓冲 ====================
#define FRAME_BUF_SIZE (64 * 1024)
static uint8_t g_frame_buf[FRAME_BUF_SIZE];
static size_t  g_frame_len = 0;
static int     g_frame_cnt = 0;
static unsigned long g_last_imu_read = 0;

CoapServer coapServer;

// ==================== 全局 IMU 数据 ====================
float ax, ay, az, gx, gy, gz;

// ==================== 前向声明 ====================
void calibrateGyro();
void initADXL345();
void initITG3205();
void readADXL345(int16_t &x, int16_t &y, int16_t &z);
void readITG3205(int16_t &x, int16_t &y, int16_t &z);
void updateImu();
void provideFrame(uint8_t** out, size_t* out_len, bool refresh, bool with_imu);
void provideImu(char* buf, size_t buf_size, size_t* out_len);
bool handleServo(int angle, char* resp, size_t resp_size, size_t* resp_len);
void initServo();
void setServoPulseUs(uint32_t pulse_us);
void setServoDirection(ServoDir dir);

// ==================== SETUP ====================
void setup() {
#if ARDUINO_USB_MODE
  disableUsbJtagKeepCdc();
#endif
  Serial.begin(115200);
  Wire.begin(GY85_SDA, GY85_SCL);

  initADXL345();
  initITG3205();
  Serial.println("保持GY85静止3秒校准...");
  delay(3000);
  calibrateGyro();
  Serial.println("GY85校准完成");

  WiFi.softAP(AP_SSID, AP_PASSWORD);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);

  IPAddress ip = WiFi.softAPIP();
  Serial.print("热点名称: "); Serial.println(AP_SSID);
  Serial.print("密码: "); Serial.println(AP_PASSWORD);
  Serial.print("CoAP 视频流: coap://"); Serial.print(ip); Serial.println("/stream  (纯JPEG)");
  Serial.print("CoAP IMU:    coap://"); Serial.print(ip); Serial.println("/imu     (1Hz)");
  Serial.print("CoAP 舵机:   coap://"); Serial.print(ip); Serial.println("/servo   (PUT angle:0=左/90=停/180=右)");

  camera_config_t config;
  config.pin_pwdn = -1;
  config.pin_reset = 2;
  config.pin_xclk = 15;
  config.pin_sscb_sda = 4;
  config.pin_sscb_scl = 5;
  config.pin_d7 = 16;
  config.pin_d6 = 17;
  config.pin_d5 = 18;
  config.pin_d4 = 12;
  config.pin_d3 = 10;
  config.pin_d2 = 8;
  config.pin_d1 = 9;
  config.pin_d0 = 11;
  config.pin_vsync = 6;
  config.pin_href = 7;
  config.pin_pclk = 13;
  config.pixel_format = PIXFORMAT_JPEG;
  config.xclk_freq_hz = 21000000;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 15;
  config.fb_count = 3;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.println("VGA 初始化失败，回退 CIF...");
    config.frame_size = FRAMESIZE_CIF;
    config.jpeg_quality = 10;
    err = esp_camera_init(&config);
  }
  if (err != ESP_OK) {
    Serial.print("摄像头失败: 0x"); Serial.println(err, HEX);
    while (1);
  }
  Serial.println("摄像头初始化成功 (VGA 640x480)");

  initServo();
  setServoDirection(SERVO_DIR_STOP);
  Serial.printf("连续旋转舵机已初始化 (GPIO%d, LEDC ch=%d, 已停止)\n",
                SERVO_PIN, SERVO_LEDC_CHANNEL);

  updateImu();
  coapServer.setFrameProvider(provideFrame);
  coapServer.setImuProvider(provideImu);
  coapServer.setServoHandler(handleServo);
  coapServer.begin();
  Serial.println("CoAP 服务器启动完成 (UDP 5683)");
}

// ==================== LOOP ====================
void loop() {
  // 视频传输期间不读 IMU，避免 I2C 占用总线
  if (!coapServer.isBusy()) {
    unsigned long now = millis();
    if (now - g_last_imu_read >= IMU_INTERVAL_MS) {
      updateImu();
      g_last_imu_read = now;
    }
  }

  for (int i = 0; i < 8; i++) {
    coapServer.loop();
  }
  int stream_ticks = coapServer.isStreamActive() ? 8 : 1;
  for (int i = 0; i < stream_ticks; i++) {
    coapServer.tickStream();
  }
}

// ==================== 舵机控制 ====================
void initServo() {
  ledc_timer_config_t tcfg = {};
  tcfg.speed_mode = SERVO_LEDC_MODE;
  tcfg.duty_resolution = (ledc_timer_bit_t)SERVO_RES_BITS;
  tcfg.timer_num = SERVO_LEDC_TIMER;
  tcfg.freq_hz = SERVO_FREQ_HZ;
  tcfg.clk_cfg = LEDC_AUTO_CLK;
  if (ledc_timer_config(&tcfg) != ESP_OK) {
    Serial.println("舵机 LEDC timer 失败");
    return;
  }

  ledc_channel_config_t ccfg = {};
  ccfg.gpio_num = SERVO_PIN;
  ccfg.speed_mode = SERVO_LEDC_MODE;
  ccfg.channel = SERVO_LEDC_CHANNEL;
  ccfg.timer_sel = SERVO_LEDC_TIMER;
  ccfg.duty = 0;
  ccfg.hpoint = 0;
  if (ledc_channel_config(&ccfg) != ESP_OK) {
    Serial.printf("舵机 LEDC GPIO%d 失败\n", SERVO_PIN);
  }
}

void setServoPulseUs(uint32_t pulse_us) {
  uint32_t max_duty = (1UL << SERVO_RES_BITS) - 1;
  uint32_t duty = (uint32_t)((uint64_t)pulse_us * max_duty / SERVO_PERIOD_US);

  ledc_set_duty(SERVO_LEDC_MODE, SERVO_LEDC_CHANNEL, duty);
  ledc_update_duty(SERVO_LEDC_MODE, SERVO_LEDC_CHANNEL);

  Serial.printf("[Servo] dir=%d pulse=%uus duty=%u\n", (int)g_servo_dir, pulse_us, duty);
}

void setServoDirection(ServoDir dir) {
  g_servo_dir = dir;
  uint32_t pulse_us = SERVO_STOP_US;
  if (dir == SERVO_DIR_CCW) pulse_us = SERVO_CCW_US;
  else if (dir == SERVO_DIR_CW) pulse_us = SERVO_CW_US;
  setServoPulseUs(pulse_us);
}

bool handleServo(int angle, char* resp, size_t resp_size, size_t* resp_len) {
  // CoAP 兼容旧协议: 0=逆时针, 90=停止, 180=顺时针
  if (angle == 0) setServoDirection(SERVO_DIR_CCW);
  else if (angle == 180) setServoDirection(SERVO_DIR_CW);
  else setServoDirection(SERVO_DIR_STOP);

  const char* dir_str = "stop";
  if (g_servo_dir == SERVO_DIR_CCW) dir_str = "left";
  else if (g_servo_dir == SERVO_DIR_CW) dir_str = "right";

  int n = snprintf(resp, resp_size,
    "{\"angle\":%d,\"dir\":\"%s\",\"ok\":true}", angle, dir_str);
  *resp_len = (n > 0 && (size_t)n < resp_size) ? (size_t)n : 0;
  return true;
}

// ==================== IMU 更新 ====================
void updateImu() {
  int16_t ax_raw, ay_raw, az_raw;
  int16_t gx_raw, gy_raw, gz_raw;

  readADXL345(ax_raw, ay_raw, az_raw);
  readITG3205(gx_raw, gy_raw, gz_raw);

  gx_raw -= gx_off;
  gy_raw -= gy_off;
  gz_raw -= gz_off;

  ax = ax_raw * ACCEL_SCALE;
  ay = ay_raw * ACCEL_SCALE;
  az = az_raw * ACCEL_SCALE;
  gx = gx_raw * GYRO_SCALE;
  gy = gy_raw * GYRO_SCALE;
  gz = gz_raw * GYRO_SCALE;
}

void provideImu(char* buf, size_t buf_size, size_t* out_len) {
  int n = snprintf(buf, buf_size,
    "{\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,\"gx\":%.4f,\"gy\":%.4f,\"gz\":%.4f}",
    ax, ay, az, gx, gy, gz);
  *out_len = (n > 0 && (size_t)n < buf_size) ? (size_t)n : 0;
}

void provideFrame(uint8_t** out, size_t* out_len, bool refresh, bool with_imu) {
  if (refresh) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      *out = nullptr;
      *out_len = 0;
      return;
    }

    if (with_imu) {
      char imu_json[128];
      int imu_len = snprintf(imu_json, sizeof(imu_json),
        "{\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,\"gx\":%.4f,\"gy\":%.4f,\"gz\":%.4f}",
        ax, ay, az, gx, gy, gz);

      size_t total = 2 + (size_t)imu_len + fb->len;
      if (total > FRAME_BUF_SIZE) {
        esp_camera_fb_return(fb);
        *out = nullptr;
        *out_len = 0;
        return;
      }

      g_frame_buf[0] = (imu_len >> 8) & 0xFF;
      g_frame_buf[1] = imu_len & 0xFF;
      memcpy(g_frame_buf + 2, imu_json, imu_len);
      memcpy(g_frame_buf + 2 + imu_len, fb->buf, fb->len);
      g_frame_len = total;
    } else {
      if (fb->len > FRAME_BUF_SIZE) {
        esp_camera_fb_return(fb);
        *out = nullptr;
        *out_len = 0;
        return;
      }
      memcpy(g_frame_buf, fb->buf, fb->len);
      g_frame_len = fb->len;
    }

    g_frame_cnt++;
    if (g_frame_cnt % 30 == 0) {
      Serial.printf("[CoAP#%d] JPEG=%uB\n", g_frame_cnt, fb->len);
    }

    esp_camera_fb_return(fb);
  }

  *out = g_frame_buf;
  *out_len = g_frame_len;
}

// ==================== GY85 函数 ====================
void calibrateGyro() {
  long sx = 0, sy = 0, sz = 0;
  int n = 100;
  for (int i = 0; i < n; i++) {
    int16_t x, y, z;
    readITG3205(x, y, z);
    sx += x; sy += y; sz += z;
    delay(5);
  }
  gx_off = sx / n;
  gy_off = sy / n;
  gz_off = sz / n;
}

void initADXL345() {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x2D); Wire.write(0x08); Wire.endTransmission();
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x31); Wire.write(0x00); Wire.endTransmission();
}

void readADXL345(int16_t &x, int16_t &y, int16_t &z) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x32); Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, 6);
  x = (int16_t)(Wire.read() | Wire.read() << 8);
  y = (int16_t)(Wire.read() | Wire.read() << 8);
  z = (int16_t)(Wire.read() | Wire.read() << 8);
}

void initITG3205() {
  Wire.beginTransmission(ITG3205_ADDR);
  Wire.write(0x3E); Wire.write(0x80); Wire.endTransmission();
  Wire.beginTransmission(ITG3205_ADDR);
  Wire.write(0x16); Wire.write(0x1B); Wire.endTransmission();
}

void readITG3205(int16_t &x, int16_t &y, int16_t &z) {
  Wire.beginTransmission(ITG3205_ADDR);
  Wire.write(0x1D); Wire.endTransmission(false);
  Wire.requestFrom(ITG3205_ADDR, 6);
  x = (int16_t)(Wire.read() << 8 | Wire.read());
  y = (int16_t)(Wire.read() << 8 | Wire.read());
  z = (int16_t)(Wire.read() << 8 | Wire.read());
}

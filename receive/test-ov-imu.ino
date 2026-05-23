#include <Wire.h>
#include <math.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ==================== GY85 配置 ====================
#define GY85_SDA 21
#define GY85_SCL 14

#define ADXL345_ADDR 0x53
#define ITG3205_ADDR 0x68

const float ACCEL_SCALE = 2.0 * 9.80665 / 32768.0;
const float GYRO_SCALE = 250.0 * (PI / 180.0) / 32768.0;

int16_t gx_off = 0, gy_off = 0, gz_off = 0;

// ==================== WiFi 热点配置 ====================
#define AP_SSID "ESP32-Camera-AP"
#define AP_PASSWORD "12345678"

// ==================== HTTP 声明 ====================
esp_err_t camera_handler(httpd_req_t *req);
esp_err_t imu_handler(httpd_req_t *req);
httpd_handle_t start_server(void);

// ==================== 全局 IMU 数据 ====================
float ax, ay, az, gx, gy, gz;

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  Wire.begin(GY85_SDA, GY85_SCL);

  // 初始化传感器
  initADXL345();
  initITG3205();
  Serial.println("保持GY85静止3秒校准...");
  delay(3000);
  calibrateGyro();
  Serial.println("GY85校准完成");

  // 开启WiFi热点
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  IPAddress ip = WiFi.softAPIP();
  Serial.print("热点名称: "); Serial.println(AP_SSID);
  Serial.print("密码: "); Serial.println(AP_PASSWORD);
  Serial.print("视频流: http://"); Serial.println(ip);
  Serial.print("IMU数据: http://"); Serial.print(ip); Serial.println("/imu");

  // 摄像头初始化
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
  config.xclk_freq_hz = 24000000;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 15;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.print("摄像头失败: 0x"); Serial.println(err, HEX);
    while (1);
  }
  Serial.println("摄像头初始化成功");

  // 启动网页服务器
  start_server();
  Serial.println("服务器启动完成！");
}

// ==================== LOOP 只负责读取传感器 ====================
void loop() {
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

  delay(40); // 25Hz
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

// ==================== 摄像头流处理 ====================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

esp_err_t camera_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  camera_fb_t *fb = NULL;

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) break;

    char buf[64];
    size_t hlen = snprintf(buf, 64, STREAM_PART, fb->len);
    httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    httpd_resp_send_chunk(req, buf, hlen);
    httpd_resp_send_chunk(req, (char*)fb->buf, fb->len);

    esp_camera_fb_return(fb);
  }
  return ESP_OK;
}

// ==================== IMU JSON接口 ====================
esp_err_t imu_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "application/json");
  char json[128];
  snprintf(json, 128,
    "{\"accel\":[%.4f,%.4f,%.4f],\"gyro\":[%.4f,%.4f,%.4f]}",
    ax, ay, az, gx, gy, gz);
  httpd_resp_send(req, json, strlen(json));
  return ESP_OK;
}

// ==================== 启动服务器 ====================
httpd_handle_t start_server(void) {
  httpd_handle_t server = NULL;
  httpd_config_t conf = HTTPD_DEFAULT_CONFIG();

  httpd_uri_t uri_cam = {"/", HTTP_GET, camera_handler, NULL};
  httpd_uri_t uri_imu = {"/imu", HTTP_GET, imu_handler, NULL};

  if (httpd_start(&server, &conf) == ESP_OK) {
    httpd_register_uri_handler(server, &uri_cam);
    httpd_register_uri_handler(server, &uri_imu);
  }
  return server;
}
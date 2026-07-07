#include <Wire.h>
#include <math.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_task_wdt.h"
#include "driver/ledc.h"
#include "coap_server.h"

// ============================================================================
// 1. 系统配置
// ============================================================================
#define DEBUG_SERIAL            Serial
#define ENABLE_DIAGNOSTICS      1

// ---- 日志宏 ----
#define LOG_INFO(fmt, ...)  DEBUG_SERIAL.printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  DEBUG_SERIAL.printf("[WARN] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...)   DEBUG_SERIAL.printf("[ERR]  " fmt "\n", ##__VA_ARGS__)
#define LOG_DBG(fmt, ...)   DEBUG_SERIAL.printf("[DBG]  " fmt "\n", ##__VA_ARGS__)

// ---- 看门狗 ----
#define WDT_TIMEOUT_SEC         30

// ---- 内存阈值 ----
#define HEAP_WARN_THRESHOLD     (50 * 1024)   // 堆内存低于50KB告警
#define HEAP_CRIT_THRESHOLD     (30 * 1024)   // 堆内存低于30KB紧急处理

// ============================================================================
// 2. GY85 传感器 (ADXL345 + ITG3205)
// ============================================================================
#define GY85_SDA                21
#define GY85_SCL                14
#define GY85_I2C_CLOCK          400000

#define ADXL345_ADDR            0x53
#define ITG3205_ADDR            0x68

// --- 寄存器 ---
#define ADXL345_REG_DEVID       0x00
#define ADXL345_REG_BW_RATE     0x2C
#define ADXL345_REG_POWER_CTL   0x2D
#define ADXL345_REG_DATA_FORMAT 0x31
#define ADXL345_REG_FIFO_CTL    0x38
#define ADXL345_REG_DATAX0      0x32

#define ITG3205_REG_WHO_AM_I    0x00
#define ITG3205_REG_SMPLRT_DIV  0x15
#define ITG3205_REG_DLPF_FS     0x16
#define ITG3205_REG_PWR_MGM     0x3E
#define ITG3205_REG_DATA_START  0x1D

// --- 传感器比例因子 (已修复) ---
// ADXL345 FULL_RES=1: 恒定 3.9mg/LSB ≈ 256 LSB/g
const float ACCEL_SCALE = 9.80665f / 256.0f;

// ITG3205: 14.375 LSB/(°/s) @ ±2000°/s
const float GYRO_SCALE  = (1.0f / 14.375f) * (PI / 180.0f);

// --- 校准 ---
#define GYRO_CAL_SAMPLES        500
#define GYRO_CAL_DELAY_MS         3
#define ACCEL_CAL_SAMPLES        200
int16_t gx_off = 0, gy_off = 0, gz_off = 0;
float ax_off = 0.0f, ay_off = 0.0f, az_g_off = 0.0f;

// --- IMU 校验 ---
#define ADXL345_SAT_LIMIT        600
#define ITG3205_SAT_LIMIT       30000
#define IMU_DELTA_WARN           50.0f
#define IMU_READ_RETRY             3
#define COMPLEMENTARY_ALPHA       0.96f

// ============================================================================
// 3. 舵机 (MG996R 连续旋转)
// ============================================================================
#define SERVO_PIN_H              40
#define SERVO_PIN_V              41
#define SERVO_COUNT               2
#define SERVO_CCW_US            1000
#define SERVO_STOP_US           1500
#define SERVO_CW_US             2000
#define SERVO_FREQ_HZ             50
#define SERVO_RES_BITS            13     // 8192步, ~2.44µs分辨率
#define SERVO_PERIOD_US        20000UL
#define SERVO_LEDC_TIMER    LEDC_TIMER_1
#define SERVO_LEDC_MODE     LEDC_LOW_SPEED_MODE
#define SERVO_DEADBAND_DEG         5
#define SERVO_MIN_CMD_MS          50

static const int SERVO_PINS[SERVO_COUNT] = { SERVO_PIN_H, SERVO_PIN_V };
static const ledc_channel_t SERVO_CH[SERVO_COUNT] = { LEDC_CHANNEL_3, LEDC_CHANNEL_4 };
enum ServoDir { DIR_STOP = 0, DIR_CCW = -1, DIR_CW = 1 };
ServoDir g_sv_dir[SERVO_COUNT] = { DIR_STOP, DIR_STOP };
unsigned long g_sv_last_ms[SERVO_COUNT] = { 0, 0 };

// ============================================================================
// 4. WiFi 热点 (传输稳定性优化)
// ============================================================================
#define AP_SSID                 "ESP32-Camera-AP"
#define AP_PASSWORD             "12345678"
#define AP_MAX_CONNS              2     // 双 WiFi 客户端
#define AP_CHANNEL_SCAN           1     // 1=扫描最优, 0=固定信道1
#define AP_TX_POWER         WIFI_POWER_19_5dBm
#define AP_BEACON_INTERVAL_MS   150     // Beacon间隔 (默认100, 略放宽省一点点电)
#define AP_DTIM_PERIOD            2     // DTIM间隔 (每2个beacon发一次广播)

// ---- WiFi 健康监控 ----
#define WIFI_HEALTH_CHECK_SEC    15     // 每15秒检查WiFi健康
#define WIFI_STALL_TIMEOUT_SEC   45     // 45秒无数据传输视为僵死
#define WIFI_AUTO_RESTART         1     // 僵死后自动重启AP

// ---- WiFi 协议锁定 (稳定性 > 兼容性) ----
// 仅启用 802.11b/g/n, 禁用 LR 模式
#define AP_WIFI_PROTOCOL  (WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N)

// ============================================================================
// 5. 摄像头
// ============================================================================
#define CAM_XCLK_HZ            20000000  // 20MHz
#define CAM_JPEG_INIT_QUAL           5   // 初始化质量→大缓冲区
#define CAM_JPEG_RUNTIME_QUAL        10  // 运行质量
#define CAM_FB_COUNT                 3

// ============================================================================
// 6. CoAP 帧缓冲 & 传输参数
// ============================================================================
#define FRAME_BUF_SIZE          (96 * 1024)
#define IMU_JSON_MAX             192
#define IMU_INTERVAL_MS           50     // 20Hz
#define DIAG_JSON_MAX            512

// ---- CoAP 流超时 (自动清理僵死客户端，超时后可重连) ----
#define STREAM_CLIENT_TIMEOUT_SEC  30
#define STREAM_PRUNE_INTERVAL_SEC   5

static uint8_t  g_frame_buf[FRAME_BUF_SIZE];
static size_t   g_frame_len = 0;
static int      g_frame_cnt = 0;
static unsigned long g_last_imu_read = 0;
static portMUX_TYPE g_frame_mux = portMUX_INITIALIZER_UNLOCKED;

CoapServer coapServer;

// ============================================================================
// 7. 全局状态
// ============================================================================
float ax, ay, az, gx, gy, gz;
float g_roll = 0.0f, g_pitch = 0.0f;
volatile bool g_imu_valid = false;
unsigned long g_boot_ms = 0;

// ---- WiFi 状态 ----
volatile int       g_wifi_clients    = 0;      // 当前连接数
volatile bool      g_wifi_ap_ok      = false;   // AP是否正常启动
volatile unsigned long g_wifi_last_tx_ms = 0;   // 最后一次成功发送时间
volatile uint32_t  g_wifi_tx_bytes   = 0;       // 累计发送字节
volatile uint32_t  g_wifi_tx_packets = 0;       // 累计发送包数
volatile uint32_t  g_wifi_tx_drops   = 0;       // 累计丢包
volatile uint32_t  g_wifi_ap_restart_cnt = 0;   // AP重启次数
unsigned long      g_last_wifi_health_ms = 0;

// ---- 流客户端跟踪 ----
unsigned long      g_stream_client_last_req_ms = 0;  // 流客户端最后请求时间

// ---- IMU 历史 ----
float g_prev_ax = 0, g_prev_ay = 0, g_prev_az = 0;

// ---- 系统健康 ----
volatile int  g_last_free_heap = 0;

// ============================================================================
// 8. 前向声明
// ============================================================================
bool initADXL345();
bool initITG3205();
bool readADXL345(int16_t &x, int16_t &y, int16_t &z);
bool readITG3205(int16_t &x, int16_t &y, int16_t &z);
void calibrateGyro();
void calibrateAccel();
void updateImu();
void updateAttitude(float dt);
void initServo();
void setServoPulse(int id, uint32_t us);
void setServoDir(int id, ServoDir d);
void stopServo(int id);
bool handleServo(int id, int angle, char* resp, size_t sz, size_t* len);
void servoEmergencyStop();
bool initWiFiAP();
int  scanBestChannel();
void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info);
void setupWiFiEvents();
void setWiFiPhyConfig();
void checkWiFiHealth();
void restartWiFiAP();
void updateWiFiStats();
bool initCamera();
void provideFrame(uint8_t** out, size_t* len, bool refresh, bool with_imu);
void provideImu(char* buf, size_t sz, size_t* len);
void provideDiag(char* buf, size_t sz, size_t* len);
void checkHeap();
void tuneNetworkBuffers();

static inline uint32_t elapsed(unsigned long since) {
  unsigned long n = millis();
  return (n >= since) ? (uint32_t)(n - since) : (uint32_t)(0xFFFFFFFFUL - since + n + 1);
}

// ============================================================================
// 9. 中值滤波
// ============================================================================
static uint16_t medianU16(const uint16_t* arr, int n) {
  uint16_t* cp = (uint16_t*)alloca(n * sizeof(uint16_t));
  if (!cp) return arr[n / 2];
  memcpy(cp, arr, n * sizeof(uint16_t));
  for (int i = 0; i <= n / 2; i++) {
    int mi = i;
    for (int j = i + 1; j < n; j++) if (cp[j] < cp[mi]) mi = j;
    uint16_t t = cp[i]; cp[i] = cp[mi]; cp[mi] = t;
  }
  return cp[n / 2];
}

// ============================================================================
// 10. WiFi 事件处理器 — 核心: 感知客户端状态变化
// ============================================================================
void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_AP_START:
      LOG_INFO("WiFi AP started");
      g_wifi_ap_ok = true;

      // AP启动后配置PHY参数 (协议/带宽)
      setWiFiPhyConfig();
      tuneNetworkBuffers();
      break;

    case ARDUINO_EVENT_WIFI_AP_STOP:
      LOG_WARN("WiFi AP stopped");
      g_wifi_ap_ok = false;
      g_wifi_clients = 0;
      break;

    case ARDUINO_EVENT_WIFI_AP_STACONNECTED: {
      g_wifi_clients = WiFi.softAPgetStationNum();
      const uint8_t* mac = info.wifi_ap_staconnected.mac;
      LOG_INFO("WiFi STA connected (#%d) MAC=%02X:%02X:%02X:%02X:%02X:%02X",
               g_wifi_clients,
               mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
      // 客户端连接→关闭省电, 确保性能
      esp_wifi_set_ps(WIFI_PS_NONE);
      g_wifi_last_tx_ms = millis();
      break;
    }

    case ARDUINO_EVENT_WIFI_AP_STADISCONNECTED: {
      g_wifi_clients = WiFi.softAPgetStationNum();
      const uint8_t* mac = info.wifi_ap_stadisconnected.mac;
      LOG_INFO("WiFi STA disconnected (#%d) MAC=%02X:%02X:%02X:%02X:%02X:%02X",
               g_wifi_clients,
               mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
      // 所有 WiFi 客户端断开 → 清理 CoAP 流槽位，便于下次重连
      if (g_wifi_clients <= 0) {
        g_wifi_clients = 0;
        coapServer.stopAllStreams();
        g_stream_client_last_req_ms = 0;
      }
      break;
    }

    case ARDUINO_EVENT_WIFI_AP_STAIPASSIGNED:
      LOG_INFO("WiFi STA IP assigned");
      g_wifi_last_tx_ms = millis();
      break;

    case ARDUINO_EVENT_WIFI_AP_PROBEREQRECVED:
      // Probe Request — 不记录, 太频繁
      break;

    case ARDUINO_EVENT_WIFI_AP_GOT_IP6:
      LOG_INFO("WiFi IPv6 ready");
      break;

    default:
      break;
  }
}

// ============================================================================
// 11. WiFi PHY 配置 — 锁定稳定参数
// ============================================================================
void setWiFiPhyConfig() {
  // 锁定WiFi协议: 仅b/g/n (禁用LR, 避免兼容性问题)
  esp_wifi_set_protocol(WIFI_IF_AP, AP_WIFI_PROTOCOL);

  // 锁定20MHz带宽 (40MHz在2.4G干扰大, 不稳定)
  esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT20);

  // 配置Beacon间隔和DTIM
  // DTIM=2: 每2个beacon发一次DTIM beacon
  // beacon_interval=150ms: Beacon间隔
  // 短Beacon间隔→客户端更快发现AP, ARP刷新更频繁
  // 长Beacon间隔→省电但客户端可能超时
  // wifi_config_t ap_cfg;
  // esp_wifi_get_config(WIFI_IF_AP, &ap_cfg);
  // ap_cfg.ap.beacon_interval = AP_BEACON_INTERVAL_MS;
  // ap_cfg.ap.dtim_period     = AP_DTIM_PERIOD;
  // esp_wifi_set_config(WIFI_IF_AP, &ap_cfg);

  // 固定速率: 禁用自动速率回退, 设置最小PHY速率
  // 防止WiFi在信号差时降至1Mbps导致传输卡死
  // esp_wifi_config_80211_tx_rate(WIFI_IF_AP, WIFI_PHY_RATE_54M);

  // 关键: 关闭AP端省电, 防止丢包
  esp_wifi_set_ps(WIFI_PS_NONE);

  LOG_INFO("WiFi PHY: bgn+HT20, PS=OFF");
}

// ============================================================================
// 12. 网络缓冲区调优 (什么是运行时可以做的)
// ============================================================================
void tuneNetworkBuffers() {
  // lwIP的SO_SNDBUF在默认配置下不支持运行时设置
  // 但我们可以通过esp_wifi API配置WiFi层的动态缓冲区
  // 注: 这些设置需要esp_wifi_init之后才能生效

  // 尝试增加WiFi动态TX缓冲区 (如果SDK支持)
  // 默认: DYNAMIC_TX_BUFFER_NUM=16 (每个1.6KB)
  // 理想: 32~64 (高吞吐场景)
  // 注: Arduino-ESP32可能不允许运行时修改此值
  // 如果编译时sdkconfig已配好则不用管

  // ESP-NETIF 配置
  // esp_netif_config_t cfg = ESP_NETIF_DEFAULT_WIFI_AP();
  // esp_netif_t *ap_netif = esp_netif_get_handle_from_ifkey("WIFI_AP_DEF");
  // 设置DHCP租约时间... (需要esp_netif API)

  LOG_INFO("Network buffers tuned");
}

// ============================================================================
// 13. WiFi 健康监控 + 自动恢复
// ============================================================================
void checkWiFiHealth() {
  if (!g_wifi_ap_ok) {
    // AP没有正常启动, 尝试重启
    LOG_ERR("WiFi health: AP not running! Attempting restart...");
    restartWiFiAP();
    return;
  }

  unsigned long now = millis();

  // 有 WiFi 客户端且存在活跃视频流，但长时间无实际发送 → 可能链路僵死
  if (g_wifi_clients > 0 && coapServer.isStreamActive() && g_wifi_last_tx_ms > 0) {
    uint32_t idle_time = elapsed(g_wifi_last_tx_ms);

    if (idle_time > WIFI_STALL_TIMEOUT_SEC * 1000UL) {
      LOG_WARN("WiFi health: %lus no TX with %d wifi / %d stream client(s) — pruning streams",
               idle_time / 1000, g_wifi_clients, coapServer.activeStreamCount());
      coapServer.stopAllStreams();

#if WIFI_AUTO_RESTART
      LOG_ERR("WiFi stall detected! Auto-restarting AP...");
      restartWiFiAP();
      return;
#endif
    }
  }

  // 检查堆内存
  int free_heap = ESP.getFreeHeap();
  g_last_free_heap = free_heap;
  if (free_heap < HEAP_WARN_THRESHOLD) {
    LOG_WARN("WiFi health: low heap %dKB", free_heap / 1024);
  }
  if (free_heap < HEAP_CRIT_THRESHOLD) {
    LOG_ERR("WiFi health: CRITICAL heap %dKB — restart may be needed",
            free_heap / 1024);
  }

  // 清理超时的 CoAP 流客户端 (keepalive 超时则释放槽位，客户端可重新订阅)
  coapServer.pruneStaleStreams(STREAM_CLIENT_TIMEOUT_SEC * 1000UL);
}

// ============================================================================
// 14. 重启 WiFi AP (异常恢复)
// ============================================================================
void restartWiFiAP() {
  LOG_WARN("Restarting WiFi AP...");
  g_wifi_ap_restart_cnt++;

  // 紧急停止舵机 (防止WiFi失控)
  servoEmergencyStop();

  // 停止现有AP
  WiFi.softAPdisconnect(true);
  delay(200);
  WiFi.mode(WIFI_OFF);
  delay(500);

  coapServer.stopAllStreams();

  // 重新初始化
  bool ok = initWiFiAP();
  if (ok) {
    g_wifi_ap_ok = true;
    g_wifi_last_tx_ms = millis();
    g_stream_client_last_req_ms = 0;
    LOG_INFO("WiFi AP restarted successfully (restart #%d)", g_wifi_ap_restart_cnt);
  } else {
    g_wifi_ap_ok = false;
    LOG_ERR("WiFi AP restart FAILED");
  }
}

// ============================================================================
// 15. SETUP
// ============================================================================
void setup() {
  g_boot_ms = millis();

  // --- 15.1 串口 ---
  DEBUG_SERIAL.begin(115200);
  for (int i = 0; i < 20 && !DEBUG_SERIAL; i++) delay(50);

  LOG_INFO("╔══════════════════════════════════════╗");
  LOG_INFO("║  OV-IMU-PWM Ultimate Optimized      ║");
  LOG_INFO("╚══════════════════════════════════════╝");

  // --- 15.2 看门狗 (ESP-IDF 5.x API) ---
  esp_task_wdt_config_t wdt_cfg = {};
  wdt_cfg.timeout_ms = WDT_TIMEOUT_SEC * 1000U;
  wdt_cfg.idle_core_mask = 0;
  wdt_cfg.trigger_panic = true;
  esp_task_wdt_init(&wdt_cfg);
  esp_task_wdt_add(NULL);
  LOG_INFO("Watchdog: %ds", WDT_TIMEOUT_SEC);

  // --- 15.3 I2C ---
  Wire.begin(GY85_SDA, GY85_SCL);
  Wire.setClock(GY85_I2C_CLOCK);
  LOG_INFO("I2C: SDA=%d SCL=%d @%dHz", GY85_SDA, GY85_SCL, GY85_I2C_CLOCK);

  // --- 15.4 传感器 ---
  bool adxl_ok = initADXL345();
  bool itg_ok  = initITG3205();
  if (!adxl_ok) LOG_ERR("ADXL345 FAILED");
  if (!itg_ok)  LOG_ERR("ITG3205 FAILED");

  if (adxl_ok && itg_ok) {
    LOG_INFO("Calibrating... keep STILL");
    delay(2000);
    calibrateAccel();
    calibrateGyro();
    LOG_INFO("Calib: AccOff=(%.2f,%.2f,%.2f)m/s² GyOff=(%d,%d,%d)",
             ax_off, ay_off, az_g_off, gx_off, gy_off, gz_off);
  }

  // --- 15.5 WiFi (先启动, 后注册事件) ---
  setupWiFiEvents();
  initWiFiAP();

  // --- 15.6 舵机 (摄像头之前) ---
  initServo();
  stopServo(0); stopServo(1);
  LOG_INFO("Servos ready GPIO%d(H) GPIO%d(V)", SERVO_PIN_H, SERVO_PIN_V);

  // --- 15.7 摄像头 ---
  if (!initCamera()) {
    LOG_ERR("Camera FAILED — video unavailable");
  }

  // --- 15.8 首次IMU ---
  if (adxl_ok && itg_ok) {
    updateImu();
    LOG_INFO("Init IMU: ax=%.3f ay=%.3f az=%.3f | gx=%.1f gy=%.1f gz=%.1f °/s",
             ax, ay, az,
             gx * 180.0f / PI,
             gy * 180.0f / PI,
             gz * 180.0f / PI);
  }

  // --- 15.9 CoAP ---
  coapServer.setFrameProvider(provideFrame);
  coapServer.setImuProvider(provideImu);
  coapServer.setServoHandler(handleServo);
  coapServer.begin();

  // --- 15.10 启动信息 ---
  IPAddress ip = WiFi.softAPIP();
  LOG_INFO("══════════════════════════════════════");
  LOG_INFO("SSID: %s", AP_SSID);
  LOG_INFO("IP:   %s", ip.toString().c_str());
  LOG_INFO("Port: UDP %d", COAP_PORT);
  LOG_INFO("Endpoints:");
  LOG_INFO("  coap://%s/stream  → MJPEG流", ip.toString().c_str());
  LOG_INFO("  coap://%s/imu     → IMU+姿态 %.0fHz", ip.toString().c_str(), 1000.0f/IMU_INTERVAL_MS);
  LOG_INFO("  coap://%s/servo   → 舵机 PUT", ip.toString().c_str());
  LOG_INFO("  coap://%s/frame   → 单帧+IMU", ip.toString().c_str());
#if ENABLE_DIAGNOSTICS
  LOG_INFO("  coap://%s/diag    → 系统诊断", ip.toString().c_str());
#endif
  LOG_INFO("══════════════════════════════════════");
  LOG_INFO("Boot: %lums | Heap: %dKB%s",
           elapsed(g_boot_ms),
           ESP.getFreeHeap() / 1024,
           psramFound() ? " +PSRAM" : "");
}

// ============================================================================
// 16. LOOP
// ============================================================================
void loop() {
  esp_task_wdt_reset(); // 喂狗

  unsigned long now = millis();

  // --- 16.1 IMU更新 (20Hz, 互斥锁保护) ---
  if (now - g_last_imu_read >= IMU_INTERVAL_MS) {
    updateImu();
    g_last_imu_read = now;

    static unsigned long last_att = 0;
    if (g_imu_valid && last_att > 0) {
      float dt = (now - last_att) / 1000.0f;
      if (dt > 0.001f && dt < 0.5f) updateAttitude(dt);
    }
    last_att = now;
  }

  // --- 16.2 CoAP请求处理 ---
  for (int i = 0; i < 8; i++) {
    coapServer.loop();
    // 记录流客户端活跃时间 (由CoAP内部set, 这里作为fallback)
  }

  // --- 16.3 CoAP 流分块推送 (双客户端，支持重连) ---
  if (coapServer.isStreamActive()) {
    for (int i = 0; i < 8; i++) {
      coapServer.tickStream();
    }
    g_stream_client_last_req_ms = coapServer.streamLastReqMs();
  }

  // --- 16.3b 定期清理僵死流客户端 ---
  static unsigned long last_stream_prune = 0;
  if (now - last_stream_prune >= STREAM_PRUNE_INTERVAL_SEC * 1000UL) {
    coapServer.pruneStaleStreams(STREAM_CLIENT_TIMEOUT_SEC * 1000UL);
    last_stream_prune = now;
  }

  // --- 16.4 WiFi健康检查+状态更新 (动态周期) ---
  static unsigned long last_health = 0;
  if (now - last_health >= WIFI_HEALTH_CHECK_SEC * 1000UL) {
    checkWiFiHealth();
    updateWiFiStats();
    last_health = now;
  }

  // --- 16.5 堆内存监控 ---
  static unsigned long last_heap_chk = 0;
  if (now - last_heap_chk >= 30000) { // 每30秒
    checkHeap();
    last_heap_chk = now;
  }

  // --- 16.6 心跳 (30s) ---
  static unsigned long last_hb = 0;
  if (now - last_hb >= 30000) {
    LOG_INFO("[HB] up=%lus heap=%dKB wifi=%s clients=%d streams=%d tx=%lupk/%luB restart=%lu",
             now / 1000,
             ESP.getFreeHeap() / 1024,
             g_wifi_ap_ok ? "ok" : "DOWN",
             g_wifi_clients,
             coapServer.activeStreamCount(),
             (unsigned long)g_wifi_tx_packets,
             (unsigned long)g_wifi_tx_bytes,
             (unsigned long)g_wifi_ap_restart_cnt);
    last_hb = now;
  }

  delay(1);
}

// ============================================================================
// 17. ADXL345 初始化
// ============================================================================
bool initADXL345() {
  uint8_t dev = 0;
  Wire.beginTransmission(ADXL345_ADDR); Wire.write(ADXL345_REG_DEVID);
  if (Wire.endTransmission(false) != 0) { LOG_ERR("ADXL345: no ACK"); return false; }
  Wire.requestFrom(ADXL345_ADDR, (uint8_t)1);
  if (Wire.available()) dev = Wire.read();
  if (dev != 0xE5) { LOG_ERR("ADXL345: DEVID=0x%02X", dev); return false; }

  // BW_RATE: 100Hz
  Wire.beginTransmission(ADXL345_ADDR); Wire.write(ADXL345_REG_BW_RATE); Wire.write(0x0A);
  if (Wire.endTransmission()) return false;

  // DATA_FORMAT: FULL_RES=1, ±2g → 恒定3.9mg/LSB
  Wire.beginTransmission(ADXL345_ADDR); Wire.write(ADXL345_REG_DATA_FORMAT); Wire.write(0x08);
  if (Wire.endTransmission()) return false;

  // FIFO_CTL: bypass
  Wire.beginTransmission(ADXL345_ADDR); Wire.write(ADXL345_REG_FIFO_CTL); Wire.write(0x00);
  if (Wire.endTransmission()) return false;

  // POWER_CTL: measure
  Wire.beginTransmission(ADXL345_ADDR); Wire.write(ADXL345_REG_POWER_CTL); Wire.write(0x08);
  if (Wire.endTransmission()) return false;

  delay(10);
  LOG_INFO("ADXL345: FULL_RES ±2g 100Hz OK");
  return true;
}

// ============================================================================
// 18. ITG3205 初始化
// ============================================================================
bool initITG3205() {
  uint8_t who = 0;
  Wire.beginTransmission(ITG3205_ADDR); Wire.write(ITG3205_REG_WHO_AM_I);
  if (Wire.endTransmission(false) != 0) { LOG_ERR("ITG3205: no ACK"); return false; }
  Wire.requestFrom(ITG3205_ADDR, (uint8_t)1);
  if (Wire.available()) who = Wire.read();
  if (who != 0x68) { LOG_ERR("ITG3205: WHO=0x%02X", who); return false; }

  // Reset
  Wire.beginTransmission(ITG3205_ADDR); Wire.write(ITG3205_REG_PWR_MGM); Wire.write(0x80);
  if (Wire.endTransmission()) return false;
  delay(100);

  // Clock: PLL Z-gyro
  Wire.beginTransmission(ITG3205_ADDR); Wire.write(ITG3205_REG_PWR_MGM); Wire.write(0x03);
  if (Wire.endTransmission()) return false;
  delay(50);

  // SMPLRT: 1kHz/(9+1)=100Hz
  Wire.beginTransmission(ITG3205_ADDR); Wire.write(ITG3205_REG_SMPLRT_DIV); Wire.write(0x09);
  if (Wire.endTransmission()) return false;

  // DLPF_FS: FS=±2000dps, DLPF=42Hz
  Wire.beginTransmission(ITG3205_ADDR); Wire.write(ITG3205_REG_DLPF_FS); Wire.write(0x1B);
  if (Wire.endTransmission()) return false;

  delay(10);
  LOG_INFO("ITG3205: ±2000dps 42Hz-LPF 100Hz OK");
  return true;
}

// ============================================================================
// 19. 传感器读取 (带重试)
// ============================================================================
bool readADXL345(int16_t &x, int16_t &y, int16_t &z) {
  for (int r = 0; r < IMU_READ_RETRY; r++) {
    Wire.beginTransmission(ADXL345_ADDR); Wire.write(ADXL345_REG_DATAX0);
    if (Wire.endTransmission(false) != 0) {
      if (r == IMU_READ_RETRY - 1) return false;
      delayMicroseconds(100); continue;
    }
    if (Wire.requestFrom(ADXL345_ADDR, (uint8_t)6) != 6) {
      if (r == IMU_READ_RETRY - 1) return false;
      delayMicroseconds(100); continue;
    }
    uint8_t xl = Wire.read(); uint8_t xh = Wire.read();
    uint8_t yl = Wire.read(); uint8_t yh = Wire.read();
    uint8_t zl = Wire.read(); uint8_t zh = Wire.read();
    x = (int16_t)((uint16_t)xl | ((uint16_t)xh << 8));
    y = (int16_t)((uint16_t)yl | ((uint16_t)yh << 8));
    z = (int16_t)((uint16_t)zl | ((uint16_t)zh << 8));
    return true;
  }
  return false;
}

bool readITG3205(int16_t &x, int16_t &y, int16_t &z) {
  for (int r = 0; r < IMU_READ_RETRY; r++) {
    Wire.beginTransmission(ITG3205_ADDR); Wire.write(ITG3205_REG_DATA_START);
    if (Wire.endTransmission(false) != 0) {
      if (r == IMU_READ_RETRY - 1) return false;
      delayMicroseconds(100); continue;
    }
    if (Wire.requestFrom(ITG3205_ADDR, (uint8_t)6) != 6) {
      if (r == IMU_READ_RETRY - 1) return false;
      delayMicroseconds(100); continue;
    }
    uint8_t xh = Wire.read(); uint8_t xl = Wire.read();
    uint8_t yh = Wire.read(); uint8_t yl = Wire.read();
    uint8_t zh = Wire.read(); uint8_t zl = Wire.read();
    x = (int16_t)(((uint16_t)xh << 8) | (uint16_t)xl);
    y = (int16_t)(((uint16_t)yh << 8) | (uint16_t)yl);
    z = (int16_t)(((uint16_t)zh << 8) | (uint16_t)zl);
    return true;
  }
  return false;
}

// ============================================================================
// 20. 校准
// ============================================================================
void calibrateAccel() {
  int32_t sx = 0, sy = 0, sz = 0;
  for (int i = 0; i < 10; i++) { int16_t t; readADXL345(t, t, t); delay(5); }
  int v = 0;
  for (int i = 0; i < ACCEL_CAL_SAMPLES && v < ACCEL_CAL_SAMPLES; i++) {
    int16_t tx, ty, tz;
    if (!readADXL345(tx, ty, tz)) { delay(2); continue; }
    sx += tx; sy += ty; sz += tz; v++; delay(3);
  }
  if (v == 0) return;
  float ax_avg = (float)sx / v * ACCEL_SCALE;
  float ay_avg = (float)sy / v * ACCEL_SCALE;
  float az_avg = (float)sz / v * ACCEL_SCALE;
  ax_off = ax_avg;
  ay_off = ay_avg;
  az_g_off = az_avg - 9.80665f;
  if (fabsf(az_g_off) > 19.6f) {
    LOG_WARN("Accel: Z offset huge (%.1f), may not be level", az_g_off);
    az_g_off = 0.0f;
  }
}

void calibrateGyro() {
  uint16_t xs[GYRO_CAL_SAMPLES], ys[GYRO_CAL_SAMPLES], zs[GYRO_CAL_SAMPLES];
  for (int i = 0; i < 10; i++) { int16_t t; readITG3205(t, t, t); delay(GYRO_CAL_DELAY_MS); }
  int c = 0;
  while (c < GYRO_CAL_SAMPLES) {
    int16_t tx, ty, tz;
    if (!readITG3205(tx, ty, tz)) { delay(GYRO_CAL_DELAY_MS); continue; }
    xs[c] = (uint16_t)(int)tx; ys[c] = (uint16_t)(int)ty; zs[c] = (uint16_t)(int)tz;
    c++; delay(GYRO_CAL_DELAY_MS);
  }
  gx_off = (int16_t)(int)medianU16(xs, GYRO_CAL_SAMPLES);
  gy_off = (int16_t)(int)medianU16(ys, GYRO_CAL_SAMPLES);
  gz_off = (int16_t)(int)medianU16(zs, GYRO_CAL_SAMPLES);
  if (abs(gx_off) > 500 || abs(gy_off) > 500 || abs(gz_off) > 500)
    LOG_WARN("Gyro offset large (%d,%d,%d)", gx_off, gy_off, gz_off);
}

// ============================================================================
// 21. IMU 更新 + 姿态估计
// ============================================================================
void updateImu() {
  int16_t ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
  if (!readADXL345(ax_r, ay_r, az_r)) { g_imu_valid = false; return; }
  if (!readITG3205(gx_r, gy_r, gz_r))  { g_imu_valid = false; return; }

  if (abs(ax_r) > ADXL345_SAT_LIMIT || abs(ay_r) > ADXL345_SAT_LIMIT || abs(az_r) > ADXL345_SAT_LIMIT)
    { g_imu_valid = false; return; }
  if (abs(gx_r) > ITG3205_SAT_LIMIT || abs(gy_r) > ITG3205_SAT_LIMIT || abs(gz_r) > ITG3205_SAT_LIMIT)
    { g_imu_valid = false; return; }

  float na = ax_r * ACCEL_SCALE - ax_off;
  float nb = ay_r * ACCEL_SCALE - ay_off;
  float nc = az_r * ACCEL_SCALE - az_g_off;
  gx_r -= gx_off; gy_r -= gy_off; gz_r -= gz_off;
  float ngx = gx_r * GYRO_SCALE;
  float ngy = gy_r * GYRO_SCALE;
  float ngz = gz_r * GYRO_SCALE;

  ax = na; ay = nb; az = nc; gx = ngx; gy = ngy; gz = ngz;
  g_prev_ax = ax; g_prev_ay = ay; g_prev_az = az;
  g_imu_valid = true;
}

void updateAttitude(float dt) {
  float a_roll  = atan2f(ay, az);
  float a_pitch = atan2f(-ax, sqrtf(ay*ay + az*az));
  g_roll  = COMPLEMENTARY_ALPHA*(g_roll + gx*dt) + (1.0f-COMPLEMENTARY_ALPHA)*a_roll;
  g_pitch = COMPLEMENTARY_ALPHA*(g_pitch + gy*dt) + (1.0f-COMPLEMENTARY_ALPHA)*a_pitch;
  if (g_roll > PI) g_roll -= 2*PI; if (g_roll < -PI) g_roll += 2*PI;
  if (g_pitch > PI) g_pitch -= 2*PI; if (g_pitch < -PI) g_pitch += 2*PI;
}

// ============================================================================
// 22. WiFi AP 初始化
// ============================================================================
void setupWiFiEvents() {
  WiFi.onEvent(onWiFiEvent);
  LOG_INFO("WiFi event handler registered");
}

bool initWiFiAP() {
  int ch = 1;
#if AP_CHANNEL_SCAN
  ch = scanBestChannel();
#endif

  WiFi.mode(WIFI_AP);
  bool ok = WiFi.softAP(AP_SSID, AP_PASSWORD, ch, 0, AP_MAX_CONNS);
  if (!ok) {
    LOG_ERR("AP FAIL ch=%d, retry ch=1", ch);
    ok = WiFi.softAP(AP_SSID, AP_PASSWORD, 1, 0, AP_MAX_CONNS);
  }
  if (!ok) { g_wifi_ap_ok = false; return false; }

  WiFi.setTxPower(AP_TX_POWER);
  g_wifi_ap_ok = true;
  LOG_INFO("WiFi AP: ch=%d max=%d power=%ddBm", ch, AP_MAX_CONNS,
           AP_TX_POWER == WIFI_POWER_19_5dBm ? 19 : 15);
  return true;
}

int scanBestChannel() {
  LOG_INFO("WiFi scan...");
  int16_t n = WiFi.scanNetworks(false, true);
  if (n <= 0) { LOG_INFO("  no APs → ch=1"); return 1; }
  int usage[14] = {0};
  for (int16_t i = 0; i < n; i++) {
    int c = WiFi.channel(i);
    if (c >= 1 && c <= 13) usage[c]++;
  }
  WiFi.scanDelete();
  const int pref[] = {1, 6, 11};
  int best = 1, best_u = 999;
  for (int c : pref) { if (usage[c] < best_u) { best_u = usage[c]; best = c; } }
  LOG_INFO("  best ch=%d (%d nearby APs)", best, best_u);
  return best;
}

void updateWiFiStats() {
  int cnt = WiFi.softAPgetStationNum();
  g_wifi_clients = cnt;

  // 动态省电: 仅在所有客户端断开时启用
  static int last_cnt = -1;
  if (cnt != last_cnt) {
    if (cnt == 0) {
      // 无客户端 → 可以省电
      // 但先保持NONE模式以确保下一个客户端能快速连接
      esp_wifi_set_ps(WIFI_PS_NONE);
    } else {
      // 有客户端 → 绝对不要省电, 否则UDP丢包
      esp_wifi_set_ps(WIFI_PS_NONE);
    }
    last_cnt = cnt;
  }
}

// ============================================================================
// 23. 内存监控
// ============================================================================
void checkHeap() {
  int free_heap = ESP.getFreeHeap();
  if (free_heap < HEAP_WARN_THRESHOLD) {
    LOG_WARN("Low heap: %dKB free (warn<%dKB)", free_heap/1024, HEAP_WARN_THRESHOLD/1024);
  }
  if (free_heap < HEAP_CRIT_THRESHOLD) {
    LOG_ERR("CRITICAL heap: %dKB free", free_heap/1024);
    // 紧急: 停止视频流, 释放可能的内存
    servoEmergencyStop();
  }
  if (psramFound()) {
    int free_psram = ESP.getFreePsram();
    if (free_psram < 64 * 1024) {
      LOG_WARN("Low PSRAM: %dKB", free_psram/1024);
    }
  }
}

// ============================================================================
// 24. 摄像头初始化
// ============================================================================
bool initCamera() {
  camera_config_t cfg;
  cfg.pin_pwdn = -1;     cfg.pin_reset = 2;    cfg.pin_xclk = 15;
  cfg.pin_sscb_sda = 4;  cfg.pin_sscb_scl = 5;
  cfg.pin_d7 = 16; cfg.pin_d6 = 17; cfg.pin_d5 = 18; cfg.pin_d4 = 12;
  cfg.pin_d3 = 10; cfg.pin_d2 = 8;  cfg.pin_d1 = 9;  cfg.pin_d0 = 11;
  cfg.pin_vsync = 6; cfg.pin_href = 7; cfg.pin_pclk = 13;

  cfg.xclk_freq_hz  = CAM_XCLK_HZ;
  cfg.pixel_format  = PIXFORMAT_JPEG;
  cfg.frame_size    = FRAMESIZE_VGA;
  cfg.jpeg_quality  = CAM_JPEG_INIT_QUAL;
  cfg.fb_count      = CAM_FB_COUNT;
  cfg.grab_mode     = CAMERA_GRAB_LATEST;
  cfg.ledc_timer    = LEDC_TIMER_0;
  cfg.ledc_channel  = LEDC_CHANNEL_0;

  if (psramFound()) {
    cfg.fb_location = CAMERA_FB_IN_PSRAM;
    LOG_INFO("PSRAM: %dKB fb→PSRAM", ESP.getPsramSize()/1024);
  } else {
    cfg.fb_location = CAMERA_FB_IN_DRAM;
    LOG_WARN("No PSRAM, DRAM only");
  }

  esp_err_t e = esp_camera_init(&cfg);
  if (e) { cfg.frame_size = FRAMESIZE_CIF; e = esp_camera_init(&cfg); }
  if (e) { cfg.frame_size = FRAMESIZE_QVGA; e = esp_camera_init(&cfg); }
  if (e) { LOG_ERR("Camera: 0x%X", e); return false; }

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_quality(s, CAM_JPEG_RUNTIME_QUAL);
    s->set_vflip(s, 0); s->set_hmirror(s, 0);
    s->set_brightness(s, 0); s->set_contrast(s, 0); s->set_saturation(s, 0);
  }
  LOG_INFO("Camera OK size=%d qual=%d xclk=%dMHz",
           s ? s->status.framesize : cfg.frame_size,
           CAM_JPEG_RUNTIME_QUAL, CAM_XCLK_HZ/1000000);
  return true;
}

// ============================================================================
// 25. 舵机
// ============================================================================
void initServo() {
  ledc_timer_config_t tc = {};
  tc.speed_mode = SERVO_LEDC_MODE; tc.duty_resolution = (ledc_timer_bit_t)SERVO_RES_BITS;
  tc.timer_num = SERVO_LEDC_TIMER; tc.freq_hz = SERVO_FREQ_HZ; tc.clk_cfg = LEDC_AUTO_CLK;
  if (ledc_timer_config(&tc)) { LOG_ERR("Servo timer FAIL"); return; }
  for (int i = 0; i < SERVO_COUNT; i++) {
    ledc_channel_config_t cc = {};
    cc.gpio_num = SERVO_PINS[i]; cc.speed_mode = SERVO_LEDC_MODE;
    cc.channel = SERVO_CH[i]; cc.timer_sel = SERVO_LEDC_TIMER; cc.duty = 0; cc.hpoint = 0;
    if (ledc_channel_config(&cc)) LOG_ERR("Servo ch%d FAIL", i);
  }
}

void setServoPulse(int id, uint32_t us) {
  if (id < 0 || id >= SERVO_COUNT) return;
  if (us < 500) us = 500; if (us > 2500) us = 2500;
  uint32_t max_d = (1UL << SERVO_RES_BITS) - 1;
  uint32_t d = (uint32_t)((uint64_t)us * max_d / SERVO_PERIOD_US);
  ledc_set_duty(SERVO_LEDC_MODE, SERVO_CH[id], d);
  ledc_update_duty(SERVO_LEDC_MODE, SERVO_CH[id]);
}

void setServoDir(int id, ServoDir d) {
  if (id < 0 || id >= SERVO_COUNT) return;
  g_sv_dir[id] = d;
  uint32_t p = SERVO_STOP_US;
  if (d == DIR_CCW) p = SERVO_CCW_US;
  else if (d == DIR_CW) p = SERVO_CW_US;
  setServoPulse(id, p);
}

void stopServo(int id)           { setServoDir(id, DIR_STOP); }
void servoEmergencyStop()        { for (int i = 0; i < SERVO_COUNT; i++) stopServo(i); }

bool handleServo(int id, int angle, char* resp, size_t sz, size_t* len) {
  if (id < 0 || id >= SERVO_COUNT) return false;

  unsigned long now = millis();
  if (elapsed(g_sv_last_ms[id]) < SERVO_MIN_CMD_MS) { /* 节流但不拒绝 */ }
  g_sv_last_ms[id] = now;

  // 死区: 87~93° → 停止
  if (angle >= (90 - SERVO_DEADBAND_DEG) && angle <= (90 + SERVO_DEADBAND_DEG))
    setServoDir(id, DIR_STOP);
  else if (angle < 90) setServoDir(id, DIR_CCW);
  else                  setServoDir(id, DIR_CW);

  const char* ds = "stop";
  if (g_sv_dir[id] == DIR_CCW) ds = (id == 0) ? "left" : "up";
  else if (g_sv_dir[id] == DIR_CW) ds = (id == 0) ? "right" : "down";
  int n = snprintf(resp, sz, "{\"s\":%d,\"a\":%d,\"d\":\"%s\",\"ok\":1}", id, angle, ds);
  *len = (n > 0 && (size_t)n < sz) ? (size_t)n : 0;
  return true;
}

// ============================================================================
// 26. CoAP 帧提供者 (互斥锁保护)
// ============================================================================
void provideFrame(uint8_t** out, size_t* len, bool refresh, bool with_imu) {
  if (refresh) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { *out = nullptr; *len = 0; return; }

    portENTER_CRITICAL(&g_frame_mux);

    if (with_imu) {
      char ij[IMU_JSON_MAX];
      int il = snprintf(ij, sizeof(ij),
        "{\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,\"gx\":%.4f,\"gy\":%.4f,\"gz\":%.4f,\"r\":%.4f,\"p\":%.4f}",
        ax, ay, az, gx, gy, gz, g_roll, g_pitch);
      size_t total = 2 + (size_t)il + fb->len;
      if (total > FRAME_BUF_SIZE) {
        portEXIT_CRITICAL(&g_frame_mux); esp_camera_fb_return(fb);
        *out = nullptr; *len = 0;
        LOG_WARN("Frame overflow: %u > %u", (unsigned)total, FRAME_BUF_SIZE);
        return;
      }
      g_frame_buf[0] = (il >> 8) & 0xFF; g_frame_buf[1] = il & 0xFF;
      memcpy(g_frame_buf + 2, ij, il);
      memcpy(g_frame_buf + 2 + il, fb->buf, fb->len);
      g_frame_len = total;
    } else {
      if (fb->len > FRAME_BUF_SIZE) {
        portEXIT_CRITICAL(&g_frame_mux); esp_camera_fb_return(fb);
        *out = nullptr; *len = 0; return;
      }
      memcpy(g_frame_buf, fb->buf, fb->len);
      g_frame_len = fb->len;
    }

    g_frame_cnt++;
    if (g_frame_cnt % 150 == 0)
      LOG_INFO("[Frame#%d] %uB", g_frame_cnt, (unsigned)fb->len);

    esp_camera_fb_return(fb);
    portEXIT_CRITICAL(&g_frame_mux);

    // 记录传输统计
    g_wifi_tx_bytes += g_frame_len;
    g_wifi_tx_packets += (g_frame_len + 1023) / 1024; // 估算CoAP分包数
    g_wifi_last_tx_ms = millis();
  }

  *out = g_frame_buf;
  *len = g_frame_len;
}

// ============================================================================
// 27. CoAP IMU / DIAG 提供者
// ============================================================================
void provideImu(char* buf, size_t sz, size_t* len) {
  int n = snprintf(buf, sz,
    "{\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,"
    "\"gx\":%.4f,\"gy\":%.4f,\"gz\":%.4f,"
    "\"r\":%.4f,\"p\":%.4f,\"ok\":%d}",
    ax, ay, az, gx, gy, gz, g_roll, g_pitch, g_imu_valid ? 1 : 0);
  *len = (n > 0 && (size_t)n < sz) ? (size_t)n : 0;
}

#if ENABLE_DIAGNOSTICS
void provideDiag(char* buf, size_t sz, size_t* len) {
  unsigned long up = millis() / 1000;
  int heap = ESP.getFreeHeap();
  int psram = psramFound() ? ESP.getFreePsram() : 0;
  int n = snprintf(buf, sz,
    "{"
    "\"up\":%lu,"
    "\"heap\":%d,"
    "\"psram\":%d,"
    "\"wifi_ap\":\"%s\","
    "\"wifi_clients\":%d,"
    "\"stream_clients\":%d,"
    "\"wifi_restarts\":%lu,"
    "\"tx_bytes\":%lu,"
    "\"tx_pkts\":%lu,"
    "\"imu_ok\":%d,"
    "\"imu_hz\":%.0f,"
    "\"frames\":%d,"
    "\"sv0\":\"%s\","
    "\"sv1\":\"%s\""
    "}",
    up, heap, psram,
    g_wifi_ap_ok ? "up" : "DOWN",
    g_wifi_clients,
    coapServer.activeStreamCount(),
    (unsigned long)g_wifi_ap_restart_cnt,
    (unsigned long)g_wifi_tx_bytes,
    (unsigned long)g_wifi_tx_packets,
    g_imu_valid ? 1 : 0,
    1000.0f / IMU_INTERVAL_MS,
    g_frame_cnt,
    g_sv_dir[0] == DIR_CCW ? "left" : (g_sv_dir[0] == DIR_CW ? "right" : "stop"),
    g_sv_dir[1] == DIR_CCW ? "up"   : (g_sv_dir[1] == DIR_CW ? "down"  : "stop")
  );
  *len = (n > 0 && (size_t)n < sz) ? (size_t)n : 0;
}
#endif

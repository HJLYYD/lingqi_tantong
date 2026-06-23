# -*- coding: utf-8 -*-
import smbus2
import time
import math
import json

# I2C总线号（进迭时空K1 I2C4）
I2C_BUS = 4

# 老版本GY-85传感器地址
ADXL345_ADDR = 0x53    # 加速度计
ITG3205_ADDR = 0x68    # 陀螺仪
HMC5883L_ADDR = 0x1E   # 磁力计

# 初始化I2C总线
bus = smbus2.SMBus(I2C_BUS)

# 初始化传感器
def init_itg3205():
    bus.write_byte_data(ITG3205_ADDR, 0x15, 0x00)
    bus.write_byte_data(ITG3205_ADDR, 0x16, 0x1E)
    bus.write_byte_data(ITG3205_ADDR, 0x17, 0x00)
    bus.write_byte_data(ITG3205_ADDR, 0x3E, 0x00)

def init_adxl345():
    bus.write_byte_data(ADXL345_ADDR, 0x31, 0x08)
    bus.write_byte_data(ADXL345_ADDR, 0x2D, 0x08)

def init_hmc5883l():
    bus.write_byte_data(HMC5883L_ADDR, 0x00, 0x70)
    bus.write_byte_data(HMC5883L_ADDR, 0x01, 0x20)
    bus.write_byte_data(HMC5883L_ADDR, 0x02, 0x00)

# 读取16位原始数据
def read_raw_data(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg+1)
    value = (high << 8) | low
    return value - 65536 if value > 32767 else value

# 读取加速度计（单位：m/s²）
def read_accel():
    x = read_raw_data(ADXL345_ADDR, 0x32) / 256.0 * 9.80665
    y = read_raw_data(ADXL345_ADDR, 0x34) / 256.0 * 9.80665
    z = read_raw_data(ADXL345_ADDR, 0x36) / 256.0 * 9.80665
    return [round(x, 4), round(y, 4), round(z, 4)]

# 读取陀螺仪（单位：rad/s）
def read_gyro():
    x = read_raw_data(ITG3205_ADDR, 0x1D) / 14.375 * math.pi / 180.0
    y = read_raw_data(ITG3205_ADDR, 0x1F) / 14.375 * math.pi / 180.0
    z = read_raw_data(ITG3205_ADDR, 0x21) / 14.375 * math.pi / 180.0
    return [round(x, 4), round(y, 4), round(z, 4)]

# 主程序
if __name__ == "__main__":
    try:
        init_itg3205()
        init_adxl345()
        
        while True:
            accel = read_accel()
            gyro = read_gyro()
            
            # 输出你要求的JSON格式
            print(json.dumps({"accel": accel, "gyro": gyro}))
            
            time.sleep(0.1)  # 可调整输出频率
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")

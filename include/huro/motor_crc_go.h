/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#ifndef HURO_MOTOR_CRC_GO_H_
#define HURO_MOTOR_CRC_GO_H_

#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/bms_cmd.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/motor_cmd.hpp"
#include <array>
#include <stdint.h>

typedef struct {
  uint8_t off; // off 0xA5
  std::array<uint8_t, 3> reserve;
} BmsCmd;

typedef struct {
  uint8_t mode; // desired working mode
  float q;      // desired angle (unit: radian)
  float dq;     // desired velocity (unit: radian/second)
  float tau;    // desired output torque (unit: N.m)
  float Kp;     // desired position stiffness (unit: N.m/rad )
  float Kd;     // desired velocity stiffness (unit: N.m/(rad/s) )
  std::array<uint32_t, 3> reserve;
} MotorCmd; // motor control

typedef struct {
  std::array<uint8_t, 2> head;
  uint8_t levelFlag;
  uint8_t frameReserve;

  std::array<uint32_t, 2> SN;
  std::array<uint32_t, 2> version;
  uint16_t bandWidth;
  std::array<MotorCmd, 20> motorCmd;
  BmsCmd bms;
  std::array<uint8_t, 40> wirelessRemote;
  std::array<uint8_t, 12> led;
  std::array<uint8_t, 2> fan;
  uint8_t gpio;
  uint32_t reserve;

  uint32_t crc;
} LowCmd;

uint32_t crc32_core(uint32_t *ptr, uint32_t len);
void get_crc(unitree_go::msg::LowCmd &msg);

#endif // HURO_MOTOR_CRC_GO_H_

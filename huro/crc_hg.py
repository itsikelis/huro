import struct

import ctypes
import os
import platform


__packFmtHGLowCmd = "<2B2x" + "B3x5fI" * 35 + "5I"


def Crc(msg):
    return __Crc32(__PackHGLowCmd(msg))


def __PackHGLowCmd(msg):
    origData = []
    origData.append(msg.mode_pr)
    origData.append(msg.mode_machine)

    for i in range(35):
        origData.append(msg.motor_cmd[i].mode)
        origData.append(msg.motor_cmd[i].q)
        origData.append(msg.motor_cmd[i].dq)
        origData.append(msg.motor_cmd[i].tau)
        origData.append(msg.motor_cmd[i].kp)
        origData.append(msg.motor_cmd[i].kd)
        origData.append(msg.motor_cmd[i].reserve)

    origData.extend(msg.reserve)
    origData.append(msg.crc)

    return __Trans(struct.pack(__packFmtHGLowCmd, *origData))


def __Trans(packData):
    calcData = []
    calcLen = (len(packData) >> 2) - 1

    for i in range(calcLen):
        d = (
            (packData[i * 4 + 3] << 24)
            | (packData[i * 4 + 2] << 16)
            | (packData[i * 4 + 1] << 8)
            | (packData[i * 4])
        )
        calcData.append(d)

    return calcData


def _crc_py(data):
    bit = 0
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7

    for i in range(len(data)):
        bit = 1 << 31
        current = data[i]

        for b in range(32):
            if crc & 0x80000000:
                crc = (crc << 1) & 0xFFFFFFFF
                crc ^= polynomial
            else:
                crc = (crc << 1) & 0xFFFFFFFF

            if current & bit:
                crc ^= polynomial

            bit >>= 1

    return crc


def __Crc32(data):
    return _crc_py(data)

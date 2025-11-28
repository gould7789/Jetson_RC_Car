# -*- coding: utf-8 -*-
"""
===============================================================================
File Name    : drive.py
Description  : [버그 수정] ESC 키 입력 시 카메라 스레드도 함께 종료되도록 수정
               - stop_flag[0] = True 추가
===============================================================================
"""
import time
import Jetson.GPIO as GPIO
import evdev
import subprocess
import os

try:
    from hw_control.input_utils import HardwareInput
except ImportError:
    from input_utils import HardwareInput

# ================= 설정 (Settings) =================
KEYBOARD_NAME = "MOSART" 

# 7단계 조향 (광각 모드: 10~170도)
SERVO_STEPS = [10, 40, 70, 90, 110, 140, 170]
SERVO_INDEX = 3  # 중앙(90도)
SERVO_MIN_DC = 5.0
SERVO_MAX_DC = 10.0

# 핀 번호
MOTOR_PWM_PIN  = 33
MOTOR_DIR_PIN1 = 29
MOTOR_DIR_PIN2 = 31
SERVO_PWM_PIN  = 32

motor_speed = 50
MOTOR_STEP  = 5

# ================= PWM 활성화 =================
def activate_jetson_pwm():
    print("[System] PWM 핀(32, 33) 활성화 중...")
    try:
        subprocess.run("which busybox", shell=True, check=True, stdout=subprocess.DEVNULL)
    except:
        os.system("apt-get install -y busybox")
    cmds = [
        "busybox devmem 0x700031fc 32 0x45",
        "busybox devmem 0x6000d504 32 0x2",
        "busybox devmem 0x70003248 32 0x46",
        "busybox devmem 0x6000d100 32 0x00"
    ]
    for c in cmds:
        os.system(c)

# ================= 초기화 =================
activate_jetson_pwm()
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup([MOTOR_PWM_PIN, MOTOR_DIR_PIN1, MOTOR_DIR_PIN2, SERVO_PWM_PIN], GPIO.OUT)

motor_pwm = GPIO.PWM(MOTOR_PWM_PIN, 1000)
servo_pwm = GPIO.PWM(SERVO_PWM_PIN, 50)
motor_pwm.start(0)
servo_pwm.start(7.5) 

# ================= 함수 =================
def set_servo_angle(angle):
    duty = SERVO_MIN_DC + (angle / 180.0) * (SERVO_MAX_DC - SERVO_MIN_DC)
    servo_pwm.ChangeDutyCycle(duty)

def set_motor(speed, direction):
    if direction == 1:
        GPIO.output(MOTOR_DIR_PIN1, GPIO.LOW)
        GPIO.output(MOTOR_DIR_PIN2, GPIO.HIGH)
        motor_pwm.ChangeDutyCycle(speed)
    elif direction == -1:
        GPIO.output(MOTOR_DIR_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR_DIR_PIN2, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(speed)
    else:
        GPIO.output(MOTOR_DIR_PIN1, GPIO.LOW)
        GPIO.output(MOTOR_DIR_PIN2, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(0)

def smooth_stop():
    set_motor(0, 0)

def get_current_state():
    return SERVO_STEPS[SERVO_INDEX], motor_speed

# ================= 메인 실행 =================
def run_drive_control(stop_flag=None):
    global SERVO_INDEX, motor_speed
    
    keyboard = HardwareInput(KEYBOARD_NAME)
    if keyboard.device is None:
        print("[System] 키보드 연결 실패.")
        return

    print(f"\n=== 🏎️  주행 준비 완료! (ESC 누르면 전체 종료) ===")
    print(" [조작] ↑:전진 | ↓:후진 | ←/→:조향 | S:중앙 | T:정지")
    
    set_servo_angle(SERVO_STEPS[SERVO_INDEX])

    try:
        for event in keyboard.read_loop():
            # 외부 종료 신호 감지
            if stop_flag is not None and stop_flag[0]: 
                break
            
            if event.type == evdev.ecodes.EV_KEY:
                val = event.value
                code = event.code
                
                if code == 103: # UP
                    if val == 1 or val == 2: set_motor(motor_speed, 1)
                    elif val == 0: smooth_stop()
                elif code == 108: # DOWN
                    if val == 1 or val == 2: set_motor(motor_speed, -1)
                    elif val == 0: smooth_stop()
                elif code == 105 and val == 1: # LEFT
                    if SERVO_INDEX > 0:
                        SERVO_INDEX -= 1
                        set_servo_angle(SERVO_STEPS[SERVO_INDEX])
                        print(f"Angle: {SERVO_STEPS[SERVO_INDEX]}")
                elif code == 106 and val == 1: # RIGHT
                    if SERVO_INDEX < len(SERVO_STEPS) - 1:
                        SERVO_INDEX += 1
                        set_servo_angle(SERVO_STEPS[SERVO_INDEX])
                        print(f"Angle: {SERVO_STEPS[SERVO_INDEX]}")
                elif code == 31 and val == 1: # S
                    SERVO_INDEX = 3
                    set_servo_angle(SERVO_STEPS[SERVO_INDEX])
                elif code == 20 and val == 1: # T
                    smooth_stop()
                elif code == 30 and val == 1: # A
                    motor_speed = min(100, motor_speed + MOTOR_STEP)
                    print(f"Speed: {motor_speed}")
                elif code == 44 and val == 1: # Z
                    motor_speed = max(0, motor_speed - MOTOR_STEP)
                    print(f"Speed: {motor_speed}")
                
                # 🔴 [여기가 수정됨] ESC(1) 누르면 전체 종료 신호 전송!
                elif code == 1 and val == 1: 
                    print("🛑 ESC 입력: 전체 시스템을 종료합니다.")
                    if stop_flag is not None:
                        stop_flag[0] = True  # 카메라한테 퇴근하라고 알림!
                    break

    except Exception as e:
        print(f"[Error] {e}")

    finally:
        set_motor(0, 0)
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("Drive thread terminated.")

if __name__ == "__main__":
    run_drive_control()
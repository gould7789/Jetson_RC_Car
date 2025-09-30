import time
import Jetson.GPIO as GPIO 
from pynput import keyboard

# --- 1. 핀 및 상수 설정 (★★ 실제 연결에 맞게 수정 필요 ★★) ---

# 모터 제어 핀 (BOARD 핀 번호 기준)
DC_MOTOR_PWM = 33    # DC 모터 속도 (PWM) 핀
DC_MOTOR_IN1 = 12    # 모터 방향 (IN1) 핀
DC_MOTOR_IN2 = 13    # 모터 방향 (IN2) 핀
SERVO_PIN = 32       # 서보 모터 조향 핀

# PWM 듀티 사이클 값 (%)
STRAIGHT_ANGLE_DC = 7.5  # 직진 (중앙) 조향 값 (5.0 ~ 10.0 사이에서 조정)
LEFT_ANGLE_DC = 5.0      # 최대 좌회전 값
RIGHT_ANGLE_DC = 10.0    # 최대 우회전 값

SPEED_FORWARD_DC = 50.0  # 기본 전진 속도 (50%)
SPEED_BACKWARD_DC = -30.0 # 기본 후진 속도 (-30%)
SPEED_STOP_DC = 0.0      # 정지 속도

# 전역 변수
current_speed_dc = 0
current_steering_dc = STRAIGHT_ANGLE_DC

# --- 2. 초기화 및 GPIO 설정 ---

def initialize_gpio():
    global pwm_speed, pwm_steer
    
    # GPIO 모드를 BOARD 핀 번호 기준으로 설정
    GPIO.setmode(GPIO.BOARD)
    
    # 모든 제어 핀을 OUTPUT으로 설정
    GPIO.setup([DC_MOTOR_PWM, DC_MOTOR_IN1, DC_MOTOR_IN2, SERVO_PIN], GPIO.OUT)
    
    # PWM 객체 생성 (주파수 50Hz)
    pwm_speed = GPIO.PWM(DC_MOTOR_PWM, 50)
    pwm_steer = GPIO.PWM(SERVO_PIN, 50)
    
    # 초기 상태: 정지 및 직진
    pwm_speed.start(SPEED_STOP_DC) 
    pwm_steer.start(STRAIGHT_ANGLE_DC)
    
    print("GPIO 초기화 완료. RC카 대기 중...")


def cleanup():
    """프로그램 종료 시 GPIO 정리 및 정지"""
    print("\nRC 제어 프로그램 종료: GPIO 정리 중...")
    set_speed(SPEED_STOP_DC)
    set_steering(STRAIGHT_ANGLE_DC)
    pwm_speed.stop()
    pwm_steer.stop()
    GPIO.cleanup()
    
# --- 3. 모터 제어 함수 ---

def set_steering(angle_dc):
    """서보 모터의 조향 각도 설정"""
    global current_steering_dc
    # 조향 범위를 안전하게 제한
    current_steering_dc = max(LEFT_ANGLE_DC, min(RIGHT_ANGLE_DC, angle_dc)) 
    pwm_steer.ChangeDutyCycle(current_steering_dc)
    print(f"Steering set to: {current_steering_dc:.2f}")

def set_speed(speed_dc):
    """DC 모터의 속도 및 방향 설정"""
    global current_speed_dc
    current_speed_dc = speed_dc
    
    speed_pwm = abs(current_speed_dc)
    
    if current_speed_dc > 0:
        # 전진 설정: IN1 (HIGH), IN2 (LOW)
        GPIO.output(DC_MOTOR_IN1, GPIO.HIGH)
        GPIO.output(DC_MOTOR_IN2, GPIO.LOW)
    elif current_speed_dc < 0:
        # 후진 설정: IN1 (LOW), IN2 (HIGH)
        GPIO.output(DC_MOTOR_IN1, GPIO.LOW)
        GPIO.output(DC_MOTOR_IN2, GPIO.HIGH)
    else:
        # 정지 설정: 모두 LOW (모터 멈춤)
        GPIO.output(DC_MOTOR_IN1, GPIO.LOW)
        GPIO.output(DC_MOTOR_IN2, GPIO.LOW)
        
    pwm_speed.ChangeDutyCycle(speed_pwm)
    print(f"Speed set to: {current_speed_dc:.1f}%")

# --- 4. 키보드 이벤트 콜백 함수 ---

def on_press(key):
    """키가 눌렸을 때 호출"""
    try:
        if key.char == 'w':
            set_speed(SPEED_FORWARD_DC) # 전진
        elif key.char == 's':
            set_speed(SPEED_BACKWARD_DC) # 후진 (음수 값 전달)
        elif key.char == 'a':
            set_steering(LEFT_ANGLE_DC) # 좌회전 (최대)
        elif key.char == 'd':
            set_steering(RIGHT_ANGLE_DC) # 우회전 (최대)
        elif key.char == 'q':
            cleanup()
            return False # 리스너 종료
            
    except AttributeError:
        # 특수 키 처리 (스페이스바는 정지)
        if key == keyboard.Key.space:
            set_speed(SPEED_STOP_DC) 
            set_steering(STRAIGHT_ANGLE_DC) 
            print("--- SPACE: EMERGENCY STOP ---")
            
def on_release(key):
    """키가 떼어졌을 때 호출"""
    try:
        # 조향 키가 떼어지면 직진으로 복귀
        if key.char in ['a', 'd']:
            set_steering(STRAIGHT_ANGLE_DC)
            
        # 속도 키가 떼어지면 정지
        elif key.char in ['w', 's']:
            set_speed(SPEED_STOP_DC)
            
    except AttributeError:
        pass


# --- 5. 메인 실행 ---
if __name__ == '__main__':
    try:
        initialize_gpio()
        
        print("\n--- RC 조종 시작 ---")
        print("W: 전진 / S: 후진 / A: 좌회전 / D: 우회전")
        print("SPACE: 정지 및 직진 / Q: 프로그램 종료")
        
        # 키보드 리스너 시작
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        listener.join()

    except KeyboardInterrupt:
        print("\n\n프로그램 강제 종료 감지.")
    finally:
        cleanup()
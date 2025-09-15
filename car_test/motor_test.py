import Jetson.GPIO as GPIO
import time

# 모터 드라이버에 연결된 GPIO 핀 번호 설정 (BCM 모드)
MOTOR_A_IN1 = 31
MOTOR_A_IN2 = 29
MOTOR_A_ENA = 33

# GPIO 모드 설정 (BCM 모드: 핀 번호 대신 GPIO 번호 사용)
GPIO.setmode(GPIO.BCM)

# 핀들을 출력으로 설정
GPIO.setup(MOTOR_A_IN1, GPIO.OUT)
GPIO.setup(MOTOR_A_IN2, GPIO.OUT)
GPIO.setup(MOTOR_A_ENA, GPIO.OUT)

# PWM(펄스 폭 변조) 속도 제어용
# 주파수는 코터에 따라 다르게 설정 가능
pwm_motor_a = GPIO.PWM(MOTOR_A_ENA, 100)
# PWM 시작 (초기 듀티 사이클: 0%)
pwm_motor_a.start(0)

# 모터를 정방향으로 회전
def motor_forward(speed):
    GPIO.output(MOTOR_A_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_A_IN2, GPIO.LOW)
    pwm_motor_a.ChangeDutyCycle(speed)
    
# 모터를 역방향으로 회전
def motor_backward(speed):
    GPIO.output(MOTOR_A_IN1, GPIO.LOW)
    GPIO.output(MOTOR_A_IN2, GPIO.HIGH)
    pwm_motor_a.ChangeDutyCycle(speed)
    
# 모터를 멈춤
def motor_stop():
    GPIO.output(MOTOR_A_IN1, GPIO.LOW)
    GPIO.output(MOTOR_A_IN2, GPIO.LOW)
    pwm_motor_a.ChangeDutyCycle(0)

try:
    print("모터 테스트 시작")
    
    # 5초간 정방향 회전 (속도 70%)
    print("정방향 5초 회전")
    motor_forward(70)
    time.sleep(5)
    
    # 2초간 정지
    print("2초 정지")
    motor_stop()
    time.sleep(2)
    
    # 5초간 역방향 회전 (속도 70%)
    print("역방향 5초 회전")
    motor_backward(70)
    time.sleep(5)
    
    # 최종 정지
    print("모터 정지")
    motor_stop()

except KeyboardInterrupt:
    print("테스트 중단")

finally:
    # GPIO 설정 초기화
    pwm_motor_a.stop()
    GPIO.cleanup()
    print("GPIO 정리 완료")
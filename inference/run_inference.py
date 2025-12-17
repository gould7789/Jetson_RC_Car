import sys
import os
import time
import cv2
import numpy as np

# -------------------------------------------------------
# [1] 경로 에러 방지 (data-collector 인식용)
# -------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

sys.path.append(root_dir)
sys.path.append(current_dir)

# data-collector 폴더를 경로에 추가
data_collector_path = os.path.join(root_dir, "data-collector")
sys.path.append(data_collector_path)

# -------------------------------------------------------
# [2] 모듈 불러오기
# -------------------------------------------------------
from preprocessor.RCPreprocessor import RCPreprocessor
from inference.engine_loader import TRTInferenceEngine

# hw_control 불러오기
import hw_control.drive as drive   
import hw_control.input_utils as input_utils

# ★ 7단계 각도 설정
ANGLE_LIST = [10, 40, 70, 90, 110, 140, 170]

def main():
    print("==========================================")
    print("   🏎️  AI AUTONOMOUS DRIVING START  🏎️   ")
    print("==========================================")
    print("[INFO] Initializing Control...")
    
    try:
        drive.activate_jetson_pwm()
    except:
        pass

    # ★ 시작 속도 50으로 강제 설정
    drive.motor_speed = 50
    print(f"[SETTING] Initial Speed set to: {drive.motor_speed}")

    current_manual_idx = 3 

    # 1. 모델 로드
    engine_path = os.path.join(root_dir, "models", "pilotnet_steering.trt")
    if not os.path.exists(engine_path):
        print(f"[ERROR] 모델 파일 없음: {engine_path}")
        return
    engine = TRTInferenceEngine(engine_path)
    print("[INFO] AI Engine Loaded.")

    # 2. 전처리기 및 카메라
    preproc = RCPreprocessor(out_size=(200, 66), crop_top_ratio=0.4)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Camera Open Failed.")
        return

    print("\n[INFO] Loop Start! (Press 'T' to STOP, 'ESC' to Quit)")
    print("⚠️  WARNING: Car will move immediately! ⚠️")
    time.sleep(1) # 1초 대기 후 출발

    try:
        while True:
            # ---------------------------
            # A. 이미지 읽기
            # ---------------------------
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # ---------------------------
            # B. 키 입력 (HardwareInput)
            # ---------------------------
            key = input_utils.get_key_nonblock()
            steering_overridden = False

            if key:
                if key == "UP":
                    drive.control_motor("forward")
                elif key == "DOWN":
                    drive.control_motor("backward")
                elif key == "LEFT":
                    current_manual_idx = max(0, current_manual_idx - 1)
                    drive.set_servo_angle(ANGLE_LIST[current_manual_idx])
                    steering_overridden = True
                elif key == "RIGHT":
                    current_manual_idx = min(len(ANGLE_LIST) - 1, current_manual_idx + 1)
                    drive.set_servo_angle(ANGLE_LIST[current_manual_idx])
                    steering_overridden = True
                elif key in ("s", "S"):
                    current_manual_idx = 3
                    drive.set_servo_angle(90)
                    steering_overridden = True
                elif key in ("a", "A"):
                    drive.motor_speed = min(100, drive.motor_speed + 5)
                    print(f"[SPEED UP] -> {drive.motor_speed}")
                    # 달리는 중이면 속도 즉시 반영
                    if drive.current_direction: drive.motor_pwm.ChangeDutyCycle(drive.motor_speed)
                elif key in ("z", "Z"):
                    drive.motor_speed = max(0, drive.motor_speed - 5)
                    print(f"[SPEED DOWN] -> {drive.motor_speed}")
                    if drive.current_direction: drive.motor_pwm.ChangeDutyCycle(drive.motor_speed)
                elif key in ("t", "T"):
                    drive.smooth_stop()
                    print("[STOP] Emergency Stop!")
                elif key in ("ESC", "CTRL_C", "q"):
                    break

            # ---------------------------
            # C. AI 추론
            # ---------------------------
            img_chw = preproc(frame)
            input_batch = img_chw[np.newaxis, ...]
            logits = engine.infer(input_batch)
            pred_idx = int(np.argmax(logits, axis=1))
            pred_angle = ANGLE_LIST[pred_idx]

            # ---------------------------
            # D. 제어 (자동 주행)
            # ---------------------------
            if not steering_overridden:
                # 1. 핸들 돌리기
                drive.set_servo_angle(pred_angle)
                current_manual_idx = pred_idx
                
                # 2. ★ [핵심] 무조건 전진 시키기! (정지상태가 아니면)
                # T를 눌러서 멈춘 상태가 아니라면 계속 전진합니다.
                # drive.py에 current_direction 변수가 있어야 함.
                # 없으면 그냥 무조건 전진 명령을 내립니다.
                drive.set_motor(drive.motor_speed, 1) 

            # 화면 출력(imshow) 삭제됨 -> 속도 향상

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        cap.release()
        drive.smooth_stop()
        print("[INFO] System Shutdown.")

if __name__ == "__main__":
    main()
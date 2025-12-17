import sys

import os

import time

import cv2

import numpy as np



# [1] 경로 에러 방지

current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.dirname(current_dir)



sys.path.append(root_dir)

sys.path.append(current_dir)



data_collector_path = os.path.join(root_dir, "data-collector")

sys.path.append(data_collector_path)



# [2] 모듈 불러오기

from preprocessor.RCPreprocessor import RCPreprocessor

from inference.engine_loader import TRTInferenceEngine



import hw_control.drive as drive   

import hw_control.input_utils as input_utils



# ★ 7단계 각도 설정

ANGLE_LIST = [10, 40, 70, 90, 110, 140, 170]



# ★ [설정] 자동 탈출 민감도 (상황에 맞춰 조절 필요)

STUCK_THRESHOLD = 4.0   # 영상 변화량이 이보다 낮으면 '멈춤'으로 간주 (낮을수록 둔감)

STUCK_TIME_LIMIT = 2.0  # 몇 초 동안 멈춰 있어야 구조 모드를 발동할지 (초)

# ========================================================



def execute_rescue():

    """

    [자동 구조 함수]

    후진 -> 정지 -> 핸들 정렬 -> 재출발

    """

    print("🚨 [AUTO-RESCUE] 끼임 감지! 자동 탈출을 시도합니다...")

    

    # 1. 일단 멈춤

    drive.smooth_stop()

    time.sleep(0.5)

    

    # 2. 핸들 반대로 꺾기 (또는 중앙)

    # 벽에 박았을 땐 중앙보다는 반대로 꺾는 게 낫지만, 일단 안전하게 중앙으로

    drive.set_servo_angle(90)

    

    # 3. 힘차게 후진 (1.5초)

    print("   <<< 후진 중...")

    drive.set_motor(70, -1) # 후진은 좀 더 세게 (70)

    time.sleep(1.5)

    

    # 4. 다시 멈춤 (반동 제거)

    drive.smooth_stop()

    time.sleep(0.5)

    

    print("🚀 [RESUME] 탈출 완료. 주행 재개!")



def main():

    print("==========================================")

    print("   🏎️  AI FULL-AUTONOMOUS DRIVING  🏎️   ")

    print("==========================================")

    

    try:

        drive.activate_jetson_pwm()

    except:

        pass



    # 초기 속도

    drive.motor_speed = 60

    print(f"[SETTING] Speed: {drive.motor_speed}")



    current_manual_idx = 3 



    # 모델 로드

    engine_path = os.path.join(root_dir, "models", "pilotnet_steering.trt")

    if not os.path.exists(engine_path):

        print(f"[ERROR] 모델 없음: {engine_path}")

        return

    engine = TRTInferenceEngine(engine_path)

    print("[INFO] AI Engine Loaded.")



    # 전처리기 & 카메라

    preproc = RCPreprocessor(out_size=(200, 66), crop_top_ratio=0.4)

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



    if not cap.isOpened():

        print("[ERROR] Camera Open Failed.")

        return



    # --- [변수 초기화] 끼임 감지용 ---

    prev_gray = None        # 이전 프레임 (흑백)

    stuck_start_time = 0.0  # 멈춰있기 시작한 시간

    is_stuck_counting = False



    print("\n[INFO] Auto Pilot Start! (1초 후 출발)")

    time.sleep(1)



    try:

        while True:

            # A. 이미지 읽기

            ret, frame = cap.read()

            if not ret:

                time.sleep(0.01)

                continue



            # -------------------------------------------------

            # ★ [핵심] 자동 끼임 감지 로직 (Visual Stuck Detection)

            # -------------------------------------------------

            # 1. 연산 속도를 위해 작게 줄이고 흑백 변환

            small_frame = cv2.resize(frame, (160, 120))

            curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            

            # 2. 이전 프레임이 있으면 비교

            if prev_gray is not None:

                # 차이 계산 (절대값 차이의 평균)

                score = np.mean(cv2.absdiff(curr_gray, prev_gray))

                

                # 3. 변화량이 너무 적은지 확인 (차가 멈춤)

                if score < STUCK_THRESHOLD:

                    if not is_stuck_counting:

                        stuck_start_time = time.time()

                        is_stuck_counting = True

                    else:

                        # 4. 일정 시간 이상 멈춰있으면 구조 실행

                        elapsed = time.time() - stuck_start_time

                        if elapsed > STUCK_TIME_LIMIT:

                            execute_rescue()

                            # 구조 후에는 초기화

                            prev_gray = None 

                            is_stuck_counting = False

                            continue # 다음 루프로 넘어감 (바로 전진하지 않게)

                else:

                    # 잘 움직이고 있으면 카운터 리셋

                    is_stuck_counting = False

            

            # 현재 프레임을 '이전 프레임'으로 저장

            prev_gray = curr_gray

            # -------------------------------------------------



            # B. 키 입력 (HardwareInput)

            key = input_utils.get_key_nonblock()

            steering_overridden = False



            if key:

                if key == "UP": drive.control_motor("forward")

                elif key == "DOWN": drive.control_motor("backward")

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

                    if drive.current_direction: drive.motor_pwm.ChangeDutyCycle(drive.motor_speed)

                elif key in ("z", "Z"):

                    drive.motor_speed = max(0, drive.motor_speed - 5)

                    if drive.current_direction: drive.motor_pwm.ChangeDutyCycle(drive.motor_speed)

                elif key in ("t", "T"):

                    drive.smooth_stop()

                    print("[STOP] Manual Stop.")

                    # 멈췄을 땐 구조 카운트도 초기화

                    is_stuck_counting = False 

                elif key in ("ESC", "CTRL_C", "q"):

                    break



            # C. AI 추론

            img_chw = preproc(frame)

            input_batch = img_chw[np.newaxis, ...]

            logits = engine.infer(input_batch)

            pred_idx = int(np.argmax(logits, axis=1))

            pred_angle = ANGLE_LIST[pred_idx]



            # D. 제어 (자동 주행)

            if not steering_overridden:

                drive.set_servo_angle(pred_angle)

                current_manual_idx = pred_idx

                # 무조건 전진

                drive.set_motor(drive.motor_speed, 1) 



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





---------------------------------------------



import evdev

from evdev import ecodes

import select



# HardwareInput 클래스 (기존 잘 되던 코드 유지)

class HardwareInput:

    def __init__(self, device_name_part="MOSART"):

        self.device = None

        self._force_connect_event2()



    def _force_connect_event3(self):

        print(f"\n[System] '/dev/input/event3' (진짜 키보드)에 강제 연결 시도...")

        try:

            # 🔴 [핵심] 사용자가 확인한 'event3'로 무조건 직진!

            self.device = evdev.InputDevice('/dev/input/event3')

            

            print(f"[SUCCESS] 연결 성공!: {self.device.name}")

            print(f"          경로: {self.device.fn}")

            

        except Exception as e:

            print(f"[ERROR] 연결 실패: {e}")

            print(" -> 혹시 젯슨을 재부팅하셨나요? 재부팅하면 번호가 바뀔 수 있습니다.")

            self.device = None



    def read_loop(self):

        if self.device:

            return self.device.read_loop()

        else:

            return []



# run_inference.py와 연결하기 위한 함수

global_hw_input = None



def get_key_nonblock():

    global global_hw_input



    # 1. 처음 실행되면 HardwareInput 객체 생성 (연결 시도)

    if global_hw_input is None:

        global_hw_input = HardwareInput()

    

    # 연결 실패했으면 False 반환

    if global_hw_input.device is None:

        return False



    # 2. 논블로킹(0초 대기)으로 입력 확인

    try:

        r, w, x = select.select([global_hw_input.device.fd], [], [], 0.0)

        if not r:

            return False # 입력 없음



        # 3. 이벤트 읽어서 변환

        for event in global_hw_input.device.read():

            if event.type == ecodes.EV_KEY and event.value == 1: # 1 = Key Press

                

                if event.code == ecodes.KEY_UP:    return "UP"

                if event.code == ecodes.KEY_DOWN:  return "DOWN"

                if event.code == ecodes.KEY_LEFT:  return "LEFT"

                if event.code == ecodes.KEY_RIGHT: return "RIGHT"

                

                if event.code == ecodes.KEY_W:     return "UP"

                if event.code == ecodes.KEY_S:     return "s"

                if event.code == ecodes.KEY_A:     return "a"

                if event.code == ecodes.KEY_Z:     return "z"

                if event.code == ecodes.KEY_T:     return "t"

                if event.code == ecodes.KEY_Q:     return "q"

                if event.code == ecodes.KEY_ESC:   return "ESC"



    except Exception as e:

        print(f"[WARN] 키보드 읽기 에러: {e}")

        return False



    return False
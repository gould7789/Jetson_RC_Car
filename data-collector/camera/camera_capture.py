import cv2
import csv
import os
import time
from datetime import datetime

def camera_capture_loop(output_dir, csv_file, image_width, image_height, save_interval, stop_flag, state_getter):
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "image_path", "servo_angle", "dc_motor_speed"])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    print(f"[Camera] 촬영 시작! (무조건 저장 / {save_interval}초 간격)")
    last_save = time.time()
    save_count = 0

    while not stop_flag[0]:
        ret, frame = cap.read()
        if not ret: break

        now = time.time()
        if now - last_save >= save_interval:
            servo_angle, motor_speed = state_getter()
            
            # ✅ 조건 없이 무조건 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}_angle{servo_angle}_speed{motor_speed}.jpg"
            image_path = os.path.join(output_dir, filename)

            cv2.imwrite(image_path, frame)
            
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, filename, servo_angle, motor_speed])
            
            last_save = now
            save_count += 1
            
            if save_count % 10 == 0:
                print(f"\r[Recording] {save_count}장 저장됨 | 각도 {servo_angle}", end="")
        
        time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()
# =============================================================================
# Description : [수정됨] 저장 주기를 0.1초(10FPS)로 단축하여 정밀한 주행 데이터 확보
# =============================================================================

import threading
import time
import os

# 모듈 불러오기 (경로에 맞춰서 사용)
from camera.camera_capture import camera_capture_loop   # 영상 캡처 모듈
import hw_control.drive as drive                        # 주행 제어 모듈

# -----------------------------------------------------------------------------
# 공통 상태 / 설정값
# -----------------------------------------------------------------------------
stop_flag = [False]

# 데이터 저장 디렉토리 및 파일 경로
OUTPUT_DIR = "dataset"
CSV_FILE = "dataset/data_labels.csv"

# -----------------------------------------------------------------------------
# 🔴 [핵심 수정] 촬영 해상도 및 프레임 저장 주기 변경
# -----------------------------------------------------------------------------
IMAGE_W, IMAGE_H = 640, 480

# 기존 0.5 -> 0.1로 변경 (1초에 10장 저장)
# 이렇게 해야 곡선 주행 시 부드러운 각도 변화를 학습할 수 있습니다.
SAVE_INTERVAL = 0.1  


# -----------------------------------------------------------------------------
# 현재 주행 상태 조회 함수
# -----------------------------------------------------------------------------
def get_state():
    """
    camera_capture_loop에서 라벨 저장 시 사용하는 콜백 함수.
    반환: (servo_angle, motor_speed)
    """
    return drive.get_current_state()


# -----------------------------------------------------------------------------
# 메인 실행부
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 저장 폴더가 없으면 미리 생성 (에러 방지)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[System] '{OUTPUT_DIR}' 폴더를 생성했습니다.")

    try:
        print(f"=== 데이터 수집 시작 (1초에 {1/SAVE_INTERVAL:.0f}장 저장) ===")
        print("=== 종료하려면 Ctrl+C를 누르세요 ===")

        # 1) 주행 제어 스레드
        drive_thread = threading.Thread(
            target=drive.run_drive_control,
            args=(stop_flag,),
            daemon=True,
        )

        # 2) 카메라 캡처 스레드
        camera_thread = threading.Thread(
            target=camera_capture_loop,
            args=(
                OUTPUT_DIR,      # 저장 폴더
                CSV_FILE,        # CSV 라벨 파일
                IMAGE_W, IMAGE_H,
                SAVE_INTERVAL,   # 0.1초
                stop_flag,       # 종료 플래그
                get_state,       # 라벨 조회
            ),
            daemon=True,
        )

        # 스레드 시작
        drive_thread.start()
        camera_thread.start()

        # 두 스레드 종료까지 대기
        drive_thread.join()
        camera_thread.join()

    except KeyboardInterrupt:
        print("\n[System] 사용자 종료 요청 (Ctrl+C)")
        stop_flag[0] = True
        
        # 안전한 종료를 위해 잠시 대기
        time.sleep(1)
        print("[System] 프로그램 종료.")
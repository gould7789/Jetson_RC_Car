import cv2
import pandas as pd
import os
import time
from datetime import datetime

# ==========================================
# 설정
# 오버샘플링을 위한 코드
# ==========================================
DATASET_DIR = "dataset"            # 이미지와 CSV가 있는 폴더
CSV_FILE = "dataset/data_labels.csv"
TARGET_ANGLES = [10, 40, 70]       # 뒤집어서 늘릴 각도 (왼쪽 -> 오른쪽으로 변환)
# 예: 10도를 뒤집으면 170도, 40->140, 70->110 데이터가 생성됨

def augment_dataset():
    # 1. CSV 파일 로드
    if not os.path.exists(CSV_FILE):
        print(f"[Error] {CSV_FILE} 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"=== 현재 데이터 수: {len(df)}개 ===")
    
    # 2. 뒤집을 대상 데이터 필터링 (왼쪽 각도들만 선택)
    target_df = df[df['servo_angle'].isin(TARGET_ANGLES)]
    print(f"=== 증강 대상(왼쪽 각도) 데이터 수: {len(target_df)}개 ===")
    
    new_rows = []
    count = 0
    
    print("=== 데이터 증강 시작 (좌우 반전 이미지 생성 중...) ===")

    for index, row in target_df.iterrows():
        # 기존 파일 정보 읽기
        org_filename = row['image_path']
        org_angle = row['servo_angle']
        speed = row['dc_motor_speed']
        
        # 전체 경로
        org_path = os.path.join(DATASET_DIR, org_filename)
        
        # 이미지 로드
        img = cv2.imread(org_path)
        if img is None:
            continue
            
        # ---------------------------------------------------
        # ⭐ 핵심: 이미지와 각도 반전
        # ---------------------------------------------------
        flipped_img = cv2.flip(img, 1)      # 이미지 좌우 반전
        new_angle = 180 - org_angle         # 각도 반전 (10->170, 70->110)
        
        # ---------------------------------------------------
        # 파일명 생성 (기존 규칙 준수)
        # ---------------------------------------------------
        # 겹치지 않게 현재 시간기반 타임스탬프 생성 (약간의 delay 필요할 수 있음)
        # 루프가 너무 빠르면 파일명이 겹칠 수 있으니 인덱스 활용하거나 microsecond 활용
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # 너무 빨라서 타임스탬프가 겹치는걸 방지하기 위해 suffix 추가
        new_filename = f"{timestamp}_aug_{index}_angle{new_angle}_speed{speed}.jpg"
        new_path = os.path.join(DATASET_DIR, new_filename)
        
        # 이미지 저장
        cv2.imwrite(new_path, flipped_img)
        
        # CSV에 추가할 정보 저장
        new_rows.append({
            "timestamp": timestamp,
            "image_path": new_filename,
            "servo_angle": new_angle,
            "dc_motor_speed": speed
        })
        
        count += 1
        if count % 100 == 0:
            print(f"\r[Progress] {count}장 생성 완료...", end="")

    print(f"\n=== 총 {count}장의 이미지가 추가 생성되었습니다. ===")

    # 3. CSV 파일 업데이트
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # 기존 데이터프레임에 병합
        final_df = pd.concat([df, new_df], ignore_index=True)
        
        # CSV 저장 (기존 파일 덮어쓰기)
        final_df.to_csv(CSV_FILE, index=False)
        print(f"=== CSV 파일 업데이트 완료! 총 데이터 수: {len(final_df)}개 ===")
        
        # 결과 확인
        print("\n[증강 후 각도 분포]")
        print(final_df['servo_angle'].value_counts().sort_index())

if __name__ == "__main__":
    augment_dataset()
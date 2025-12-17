# 파일 복사를 진행하는 오버샘플링 코드
import pandas as pd
import os
import shutil
import time

CSV_FILE = "dataset/data_labels.csv"
DATA_DIR = "dataset"

def balance():
    df = pd.read_csv(CSV_FILE)
    
    # 각도별 개수 세기
    counts = df['servo_angle'].value_counts()
    max_count = counts.max() # 제일 많은 각도 기준 (예: 9000개)
    
    print("=== 현재 데이터 분포 ===")
    print(counts)
    print(f"목표 개수: 각도당 {max_count}개로 맞춤\n")

    new_rows = []
    
    # 각 각도별로 부족한 만큼 복사
    for angle in counts.index:
        current_count = counts[angle]
        needed = max_count - current_count
        
        if needed <= 0: continue
        
        print(f"Angle {angle}: {current_count}개 -> {needed}개 추가 생성 중...")
        
        # 부족한 만큼 랜덤 샘플링 (복원 추출)
        samples = df[df['servo_angle'] == angle].sample(n=needed, replace=True)
        
        for idx, row in samples.iterrows():
            org_path = os.path.join(DATA_DIR, row['image_path'])
            
            # 새 파일명 생성 (원본이름_copy_시간.jpg)
            timestamp = str(time.time()).replace('.', '')[-6:]
            new_filename = row['image_path'].replace('.jpg', f'_copy_{timestamp}_{idx}.jpg')
            new_path = os.path.join(DATA_DIR, new_filename)
            
            # 파일 복사
            shutil.copy(org_path, new_path)
            
            # 리스트에 정보 추가
            new_row = row.copy()
            new_row['image_path'] = new_filename
            new_rows.append(new_row)

    # CSV 저장
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([df, new_df], ignore_index=True)
        final_df.to_csv(CSV_FILE, index=False)
        print("\n=== 오버샘플링 완료! ===")
        print(final_df['servo_angle'].value_counts())
    else:
        print("이미 균형이 맞아서 할 게 없습니다.")

if __name__ == "__main__":
    balance()
# 오버샘플링 코드
import pandas as pd
import os
import shutil
import time
from tqdm import tqdm

# ================= 설정 =================
CSV_FILE = "dataset/data_labels.csv"
DATA_DIR = "dataset"

# [핵심] 각도별 목표 개수 설정 (코너 집중형 전략)
# 딕셔너리 형태: { 각도 : 목표개수 }
TARGET_COUNTS = {
    10: 4500,   # 급커브 (좌) -> 부족하므로 복사됨
    40: 4500,   # 커브 (좌)
    70: 3500,   # 완만 (좌)
    90: 2000,   # 직진 -> 많으므로 CSV에서 줄임 (파일삭제 X)
    110: 3500,  # 완만 (우)
    140: 4500,  # 커브 (우)
    170: 4500   # 급커브 (우)
}

def balance_data():
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] {CSV_FILE} 파일이 없습니다.")
        return

    # 1. 원본 CSV 읽기
    df = pd.read_csv(CSV_FILE)
    print(f"=== 원본 데이터 개수: {len(df)}개 ===")
    
    final_rows = [] # 최종적으로 저장될 리스트
    
    # 데이터에 존재하는 모든 각도 확인
    all_angles = sorted(df['servo_angle'].unique())
    
    print("\n[작업 시작] 데이터 밸런싱 진행 중...")

    for angle in all_angles:
        # 해당 각도의 데이터만 추출
        angle_data = df[df['servo_angle'] == angle]
        current_count = len(angle_data)
        
        # 설정된 목표가 없으면 원래 개수 유지
        target_count = TARGET_COUNTS.get(angle, current_count)

        print(f"Angle {angle}: 현재 {current_count}장 -> 목표 {target_count}장 ", end="")

        # ==========================================
        # CASE 1: 데이터가 너무 많을 때 (Undersampling - 직진 등)
        # ==========================================
        if current_count > target_count:
            print(f"[축소] {current_count - target_count}장 제외")
            # 랜덤하게 목표 개수만큼만 뽑아서 리스트에 추가 (파일 삭제 안 함)
            sampled = angle_data.sample(n=target_count, replace=False, random_state=42)
            final_rows.extend(sampled.to_dict('records'))

        # ==========================================
        # CASE 2: 데이터가 부족할 때 (Oversampling - 코너링)
        # ==========================================
        elif current_count < target_count:
            needed = target_count - current_count
            print(f"[복사] {needed}장 파일 생성 중...")
            
            # 1) 일단 원본 데이터는 다 넣음
            final_rows.extend(angle_data.to_dict('records'))
            
            # 2) 부족한 만큼 랜덤 샘플링 (중복 허용)
            samples = angle_data.sample(n=needed, replace=True, random_state=42)
            
            new_added_rows = []
            
            # 진행률 표시바 (tqdm)
            for idx, (_, row) in tqdm(enumerate(samples.iterrows()), total=samples.shape[0]):
                original_filename = os.path.basename(row['image_path'])
                full_org_path = os.path.join(DATA_DIR, original_filename)
                
                if not os.path.exists(full_org_path):
                    continue
                
                # 새 파일명 생성 (중복 방지를 위해 인덱스 추가)
                timestamp = str(time.time()).replace('.', '')[-6:]
                speed = row['dc_motor_speed']
                current_time_str = time.strftime("%Y%m%d_%H%M%S")
                
                # 파일명: 날짜_시간_고유번호_bias_각도_속도.jpg
                new_filename = f"{current_time_str}_{timestamp}_{idx}_bias_angle{angle}_speed{speed}.jpg"
                new_full_path = os.path.join(DATA_DIR, new_filename)
                
                try:
                    # 실제 파일 복사
                    shutil.copy(full_org_path, new_full_path)
                    
                    # 리스트 추가
                    new_row = row.copy()
                    new_row['image_path'] = new_filename
                    new_row['timestamp'] = f"{current_time_str}_{timestamp}"
                    new_added_rows.append(new_row)
                    
                except Exception as e:
                    print(f"에러 발생: {e}")

            final_rows.extend(new_added_rows)
            
        # ==========================================
        # CASE 3: 딱 맞을 때
        # ==========================================
        else:
            print("[유지]")
            final_rows.extend(angle_data.to_dict('records'))

    # 3. 최종 CSV 저장
    new_df = pd.DataFrame(final_rows)
    
    # 백업 파일 생성
    shutil.copy(CSV_FILE, CSV_FILE + ".bak_before_balance")
    new_df.to_csv(CSV_FILE, index=False)
    
    print("\n========================================")
    print(f"[완료] 총 데이터 개수: {len(df)} -> {len(new_df)}")
    print(f"원본 CSV는 {CSV_FILE}.bak_before_balance 로 백업되었습니다.")
    print("========================================")
    print(new_df['servo_angle'].value_counts().sort_index())

if __name__ == "__main__":
    balance_data()
# 특정 데이터만 골라서 오버샘플링
import pandas as pd
import shutil
import os
import time

CSV_FILE = "dataset/data_labels.csv"
DATA_DIR = "dataset"
FOCUS_DIR = "dataset/focus"
BOOST_FACTOR = 800  # 20배 뻥튀기

def boost_focus_images():
    if not os.path.exists(FOCUS_DIR):
        print("focus 폴더가 없습니다.")
        return

    focus_files = [f for f in os.listdir(FOCUS_DIR) if f.endswith('.jpg')]
    if not focus_files:
        print("[경고] focus 폴더가 비어있습니다!")
        return

    print(f"=== [2단계] 문제 이미지 {len(focus_files)}장을 {BOOST_FACTOR}배로 증식합니다 ===")
    
    df = pd.read_csv(CSV_FILE)
    new_rows = []
    
    # 원본 이미지가 CSV 어디에 있는지 찾기 위함
    df_lookup = df.set_index('image_path', drop=False)

    for target_file in focus_files:
        # 파일 경로 확인
        src_path = os.path.join(DATA_DIR, target_file)
        if not os.path.exists(src_path):
            src_path = os.path.join(FOCUS_DIR, target_file)
            
        # CSV 정보 가져오기 (없으면 대략 생성)
        if target_file in df_lookup.index:
            row = df_lookup.loc[target_file]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
        else:
            # CSV에 없으면 파일명에서 각도 추론
            import re
            match = re.search(r"angle(\d+)", target_file)
            angle = int(match.group(1)) if match else 10 # 기본 10도
            row = {"servo_angle": angle, "dc_motor_speed": 50, "timestamp": 0}

        for i in range(BOOST_FACTOR):
            timestamp = str(time.time()).replace('.', '')[-6:]
            new_filename = f"boost_{timestamp}_{i}_{target_file}"
            dst_path = os.path.join(DATA_DIR, new_filename)
            
            shutil.copy(src_path, dst_path)
            
            new_row = row.copy()
            new_row['image_path'] = new_filename
            new_rows.append(new_row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([df, new_df], ignore_index=True)
        final_df.to_csv(CSV_FILE, index=False)
        print(f"=== 증식 완료! 총 {len(final_df)}개로 증가함 ===")

if __name__ == "__main__":
    boost_focus_images()
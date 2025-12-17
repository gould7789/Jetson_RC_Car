# csv 파일 복구 코드
import os
import pandas as pd
import re

DATASET_DIR = "dataset"
CSV_FILE = "dataset/data_labels.csv"

def rebuild_csv_final():
    if not os.path.exists(DATASET_DIR):
        print(f"[오류] '{DATASET_DIR}' 폴더가 없습니다!")
        return

    print("=== [1단계] 이미지 파일 스캔 및 CSV 복구 시작 ===")
    files = os.listdir(DATASET_DIR)
    jpg_files = [f for f in files if f.endswith('.jpg')]
    print(f"--> 발견된 이미지: {len(jpg_files)}장")

    data_list = []
    pattern = re.compile(r"angle(\d+)")

    for filename in jpg_files:
        match = pattern.search(filename)
        if match:
            angle = int(match.group(1))
            timestamp = filename.split('_')[0]
            data_list.append({
                "timestamp": timestamp,
                "image_path": filename,
                "servo_angle": angle,
                "dc_motor_speed": 50
            })

    df = pd.DataFrame(data_list)
    df = df.sort_values(by="image_path")
    df.to_csv(CSV_FILE, index=False)
    
    print(f"=== 복구 완료: 총 {len(df)}개 ===")
    print(df['servo_angle'].value_counts().sort_index())

if __name__ == "__main__":
    rebuild_csv_final()
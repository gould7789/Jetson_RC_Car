import shutil
import os
import pandas as pd

CSV_FILE = "dataset/data_labels.csv"
BACKUP_FILE = "dataset/data_labels.csv.bak_prune"

def restore_data():
    # 1. 백업 파일이 있는지 확인
    if os.path.exists(BACKUP_FILE):
        print(f"[발견] 백업 파일({BACKUP_FILE})이 있습니다.")
        
        # 2. 복구 실행 (덮어쓰기)
        shutil.copy(BACKUP_FILE, CSV_FILE)
        print("✅ 원본 파일로 복구했습니다!")
        
        # 3. 개수 확인
        df = pd.read_csv(CSV_FILE)
        print(f"--> 현재 데이터 개수: {len(df)}개")
        print("--> (약 29,000개 정도라면 원본 복구 성공입니다.)")
        
    else:
        print("🚨 비상! 백업 파일을 찾을 수 없습니다.")
        print("dataset 폴더 안에 '.bak_prune'으로 끝나는 파일이 있는지 직접 확인해보세요.")

if __name__ == "__main__":
    restore_data()
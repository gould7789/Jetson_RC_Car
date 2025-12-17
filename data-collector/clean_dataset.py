import pandas as pd
import os

CSV_FILE = "dataset/data_labels.csv"
DATA_DIR = "dataset"

# 오버샘플링 후 되돌리는 코드
def rollback_dataset():
    if not os.path.exists(CSV_FILE):
        print("CSV 파일이 없습니다.")
        return

    # 1. CSV 읽기
    df = pd.read_csv(CSV_FILE)
    print(f"현재 데이터 개수: {len(df)}")

    # 2. '_aug_' 가 들어간 행 찾기 (삭제 대상)
    aug_mask = df['image_path'].str.contains('_aug_')
    files_to_delete = df[aug_mask]['image_path'].tolist()
    
    print(f"삭제할 증강 데이터 개수: {len(files_to_delete)}")
    
    # 3. 실제 파일 삭제
    count = 0
    for fname in files_to_delete:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            os.remove(path)
            count += 1
    print(f"실제 파일 삭제 완료: {count}장")

    # 4. CSV에서 행 삭제 및 저장
    df_clean = df[~aug_mask]
    df_clean.to_csv(CSV_FILE, index=False)
    
    print(f"CSV 복구 완료. 남은 데이터 개수: {len(df_clean)}")
    print("이제 다시 학습을 돌리시면 '순정 상태' 모델이 나옵니다.")

if __name__ == "__main__":
    rollback_dataset()
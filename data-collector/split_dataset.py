# 층화 추출 코드
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ================= 설정 =================
CSV_FILE = "dataset/data_labels.csv"
OUTPUT_DIR = "dataset"

def split_data():
    if not os.path.exists(CSV_FILE):
        print("[오류] CSV 파일이 없습니다.")
        return

    # 1. 데이터 읽기
    df = pd.read_csv(CSV_FILE)
    print(f"=== [분할 시작] 총 {len(df)}개 데이터를 7:2:1로 나눕니다 ===")

    # -------------------------------------------------------
    # 1단계: 전체에서 테스트셋(10%) 분리
    # stratify=df['servo_angle'] -> 각도별 비율을 유지하며 분리
    # -------------------------------------------------------
    df_rest, df_test = train_test_split(
        df, 
        test_size=0.1,          # 전체의 10%
        stratify=df['servo_angle'], 
        random_state=42
    )

    # -------------------------------------------------------
    # 2단계: 남은 것(90%)에서 검증셋(20%) 분리
    # 전체의 20%를 가져와야 하므로, 남은 90% 기준으로는 2/9 비율임
    # -------------------------------------------------------
    val_ratio = 0.2 / 0.9
    
    df_train, df_val = train_test_split(
        df_rest, 
        test_size=val_ratio, 
        stratify=df_rest['servo_angle'], 
        random_state=42
    )

    # 저장
    df_train.to_csv(os.path.join(OUTPUT_DIR, "data_train.csv"), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, "data_val.csv"), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "data_test.csv"), index=False)

    print("\n========================================")
    print("           데이터 분할 완료!            ")
    print("========================================")
    print(f"[1] 학습(Train) : {len(df_train)}장 (약 70%) -> data_train.csv")
    print(f"[2] 검증(Val)   : {len(df_val)}장 (약 20%) -> data_val.csv")
    print(f"[3] 테스트(Test): {len(df_test)}장 (약 10%) -> data_test.csv")
    
    print("\n[확인] 학습 데이터(Train) 내부의 각도 분포:")
    print(df_train['servo_angle'].value_counts().sort_index())
    print("--> 위 분포가 아까 본 '좌회전 몰빵' 비율과 똑같은지 확인하세요.")

if __name__ == "__main__":
    split_data()
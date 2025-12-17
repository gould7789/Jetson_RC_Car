# csv에서 특정 데이터만 지움
import pandas as pd
import shutil
import os

# ================= 설정 =================
CSV_FILE = "dataset/data_labels.csv"

# 각도별 목표 개수 (Limit)
# -1은 "건드리지 마라(전부 유지)"라는 뜻입니다.
TARGET_COUNTS = {
    10: -1,   # 좌회전: 건드리지 않음 (현재 5002개 유지)
    40: -1, # 완만 좌회전: 조금 줄임
    70: -1, # 완만 좌회전: 조금 줄임
    90: 2500, # ★ 직진: 과감하게 줄여서 10도를 돋보이게 함
    110: -1,
    140: -1,
    170: -1
}

def prune_dataset():
    if not os.path.exists(CSV_FILE):
        print("CSV 파일이 없습니다.")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"=== [정리 전] 총 {len(df)}개 ===")
    print(df['servo_angle'].value_counts().sort_index())

    final_df_list = []

    print("\n[작업 시작] 데이터를 솎아냅니다...")

    for angle in sorted(df['servo_angle'].unique()):
        angle_data = df[df['servo_angle'] == angle]
        current_count = len(angle_data)
        
        # 설정된 목표 개수 가져오기 (없으면 그대로 유지)
        limit = TARGET_COUNTS.get(angle, -1)
        
        if limit == -1 or current_count <= limit:
            # 목표보다 적거나 제한 없으면 -> 전부 가져감
            final_df_list.append(angle_data)
            print(f" -> {angle}도: {current_count}장 (유지)")
        else:
            # 목표보다 많으면 -> 랜덤하게 솎아냄
            pruned_data = angle_data.sample(n=limit, replace=False)
            final_df_list.append(pruned_data)
            print(f" -> {angle}도: {current_count}장 => {limit}장으로 축소 (삭제: {current_count - limit}장)")

    # 합치기
    final_df = pd.concat(final_df_list, ignore_index=True)
    
    # 백업 및 저장
    shutil.copy(CSV_FILE, CSV_FILE + ".bak_prune")
    final_df.to_csv(CSV_FILE, index=False)

    print("\n========================================")
    print(f"=== [정리 완료] 총 {len(final_df)}개 ===")
    print("========================================")
    print(final_df['servo_angle'].value_counts().sort_index())
    print("\n[Tip] 이제 10도가 90도보다 2배 더 많아졌습니다!")

if __name__ == "__main__":
    prune_dataset()
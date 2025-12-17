import sys
import os
import cv2
import numpy as np
import glob

# [1] 경로 설정
# 현재 이 파일(save_all_preprocessed.py)이 있는 폴더를 루트로 잡습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir 

sys.path.append(root_dir)

# 전처리기 불러오기
from preprocessor.RCPreprocessor import RCPreprocessor

def process_all():
    # 1. 원본 이미지가 있는 폴더
    # (Lane_Keeping_Car/data-collector/dataset)
    input_dir = os.path.join(root_dir, "data-collector", "dataset")
    
    # 2. 전처리된 이미지를 저장할 '새로운' 폴더
    # (Lane_Keeping_Car/data-collector/dataset_preprocessed)
    output_dir = os.path.join(root_dir, "data-collector", "dataset_preprocessed")

    print(f"[시작] 원본 경로: {input_dir}")
    print(f"[목표] 저장 경로: {output_dir}")

    # 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # 전처리기 설정 (학습 때와 100% 동일하게)
    preproc = RCPreprocessor(out_size=(200, 66), crop_top_ratio=0.4)

    # 모든 jpg 파일 찾기
    search_path = os.path.join(input_dir, "*.jpg")
    image_paths = glob.glob(search_path)
    
    total_count = len(image_paths)
    print(f"[INFO] 총 {total_count}장의 이미지를 발견했습니다.")

    if total_count == 0:
        print("[ERROR] 이미지가 없습니다! 경로를 다시 확인해주세요.")
        return

    # 루프 돌면서 변환 및 저장
    count = 0
    print("변환 시작...")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)

        # (1) 읽기
        original = cv2.imread(img_path)
        if original is None:
            continue

        # (2) 전처리 수행
        processed_tensor = preproc(original)

        # (3) 저장용 이미지로 복구 (Tensor -> BGR Image)
        viz_img = np.transpose(processed_tensor, (1, 2, 0))
        viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
        viz_img = (viz_img * 255.0).astype(np.uint8)

        # (4) 저장
        cv2.imwrite(save_path, viz_img)

        count += 1
        if count % 100 == 0:
            print(f"진행 중... ({count}/{total_count})")

    print("------------------------------------------------")
    print(f"[완료] 총 {count}장 저장 완료!")
    print(f"[확인] 저장된 폴더 위치: {output_dir}")

if __name__ == "__main__":
    process_all()
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessor.RCPreprocessor import RCPreprocessor
from preprocessor.RCAugmentor import RCAugmentor

class RCDataset(Dataset):
    """
    [수정됨] 외부에서 이미 분할된 CSV를 읽도록 변경됨.
    split: 'train' | 'val' | 'test' (데이터 분할용이 아니라, 증강 여부 판단용으로만 사용)
    """
    def __init__(self,
                 csv_filename,
                 root,
                 preprocessor: RCPreprocessor,
                 augmentor: RCAugmentor = None,
                 split: str = "train",  # 여기서는 단순히 '역할'만 표시함
                 shuffle: bool = True,
                 random_seed: int = 42):

        self.image_root = root
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.split = split

        csv_path = os.path.join(root, csv_filename)
        
        # 1. CSV 파일 읽기
        # 이미 밖에서 잘라준 파일(data_train.csv 등)이 들어오므로
        # 여기서는 통째로 다 읽습니다.
        self.df = pd.read_csv(csv_path)

        # 2. 셔플 (학습 효율을 위해 순서 섞기)
        if shuffle:
            self.df = self.df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

        # ------------------------------------------------------
        # [삭제됨] 내부에서 7:2:1로 자르는 로직 삭제
        # 이유: 이미 잘린 파일이 들어오기 때문
        # ------------------------------------------------------

        # ------------------------------------------------------
        # angle -> class index 매핑
        # (주의: train/val/test 파일에 모든 각도가 다 들어있어야 에러가 안 남)
        # ------------------------------------------------------
        # 안전하게 하드코딩으로 고정 (우리는 7개 각도를 쓴다는 걸 아니까요)
        self.angles = [10, 40, 70, 90, 110, 140, 170]
        self.angle_to_idx = {a: i for i, a in enumerate(self.angles)}

        # (디버깅용 출력)
        if split == "train":
            print(f"[RCDataset:{split}] Loaded {len(self.df)} samples from {csv_filename}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["image_path"]
        
        # CSV 컬럼명 안전장치 (가끔 공백이 들어가는 경우 대비)
        angle_col = "servo_angle" if "servo_angle" in row else row.keys()[2] 
        angle = int(row[angle_col])
        
        img_path = os.path.join(self.image_root, rel_path)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # 안전장치: 이미지 로드 실패 시 검정 화면 (크기는 전처리기 입력 사이즈에 맞춤)
            # 보통 원본 이미지가 640x480이나 320x240일 텐데, 적당히 만듭니다.
            img_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
            print(f"[Warning] Image not found: {img_path}")

        # Train일 때만 증강 적용
        if self.split == "train" and self.augmentor is not None:
            img_bgr, angle = self.augmentor(img_bgr, angle)

        img_chw = self.preprocessor(img_bgr)
        img_tensor = torch.from_numpy(img_chw).float()

        # 라벨 변환
        if angle in self.angle_to_idx:
            label = self.angle_to_idx[angle]
        else:
            # 혹시 이상한 각도가 들어오면 가장 가까운 각도로 매핑하거나 0(10도)으로 처리
            # 여기서는 안전하게 0번 인덱스로 처리
            label = 0 

        return img_tensor, label

if __name__ == "__main__":
    pass
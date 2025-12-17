import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
import re
import cv2

# ==========================================
# ★ [핵심 수정] 학습할 때 썼던 각도들을 여기에 직접 적으세요!
# ==========================================
FIXED_CLASSES = [10, 40, 70, 90, 110, 140, 170]  # <--- 본인의 각도 값

# ==========================================
# [경로 설정]
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)                  

sys.path.append(root_dir)    
sys.path.append(current_dir) 

try:
    from training.model import PilotNet  
    from preprocessor.RCPreprocessor import RCPreprocessor
except ImportError as e:
    print(f"[오류] 모듈 로드 실패: {e}")
    sys.exit()

# 경로 설정
model_dir = os.path.join(root_dir, "models")
test_data_dir = os.path.join(root_dir, "data-collector", "dataset")

# 파일 목록 (3개)
model_filenames = [
    'pilotnet_steering_20251202_200510_best.pth',
    'pilotnet_steering_20251201_193800_best.pth',
    'pilotnet_steering_20251202_165006_best.pth'
]
model_files = [os.path.join(model_dir, f) for f in model_filenames]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 데이터셋 클래스 (수정됨)
# ==========================================
class TestDataset(Dataset):
    def __init__(self, root_dir, preprocessor):
        self.root_dir = root_dir
        self.preprocessor = preprocessor
        self.image_files = []
        self.labels = []
        
        # ★ 강제로 지정한 클래스 사용
        self.classes = sorted(FIXED_CLASSES)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"[INFO] 강제 지정된 클래스: {self.classes}")
        print(f"[INFO] 클래스 번호 매핑: {self.class_to_idx}")

        all_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        pattern = re.compile(r"_angle(\d+)_speed")
        
        valid_count = 0
        for f in all_files:
            match = pattern.search(f)
            if match:
                angle = int(match.group(1))
                if angle in self.class_to_idx:
                    self.image_files.append(f)
                    self.labels.append(self.class_to_idx[angle])
                    valid_count += 1
        
        print(f"[INFO] 전체 파일 중 {valid_count}장을 평가에 사용합니다.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 1. 이미지 읽기
        image = cv2.imread(img_path)
        
        # [중요] RGB 변환 (학습때와 동일하게 맞춤)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        if image is None:
            return torch.zeros((3, 66, 200)), torch.tensor(0), img_name

        # 2. 전처리
        try:
            processed_image = self.preprocessor.process(image)
        except:
            processed_image = cv2.resize(image, (200, 66))

        # 3. 정규화 및 텐서 변환
        if isinstance(processed_image, np.ndarray):
            # ★★★ [핵심] 정수(0~255)를 실수(0.0~1.0)로 변환 ★★★
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # 텐서로 변환
            processed_image = torch.from_numpy(processed_image).float()
            
            # (H, W, C) -> (C, H, W) 채널 순서 변경
            if processed_image.shape[-1] == 3:
                processed_image = processed_image.permute(2, 0, 1)

        # ★★★ 이 부분이 빠져서 에러가 났던 겁니다! ★★★
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return processed_image, label, img_name

# ==========================================
# 실행 함수
# ==========================================
def evaluate():
    preproc = RCPreprocessor(out_size=(200, 66), crop_top_ratio=0.4)
    test_dataset = TestDataset(test_data_dir, preproc)
    
    # 샘플 테스트 (빠른 확인용 100개)
    if len(test_dataset) > 100:
        indices = list(range(100))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_classes = len(FIXED_CLASSES)

    for pth_path in model_files:
        filename_only = os.path.basename(pth_path)
        if not os.path.exists(pth_path): continue
            
        print(f"\n--- [{filename_only}] 디버깅 시작 ---")
        model = PilotNet(num_classes=num_classes, input_shape=(3, 66, 200)).to(device)
        
        try:
            checkpoint = torch.load(pth_path, map_location=device)
            model.load_state_dict(checkpoint)
        except:
            continue
            
        model.eval()
        
        correct = 0
        total = 0
        
        print(f"| {'파일명':^30} | {'정답':^5} | {'예측':^5} | {'결과':^5} |")
        print("-" * 60)
        
        with torch.no_grad():
            for i, (images, labels, fname) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                label_idx = labels.item()
                pred_idx = preds.item()
                is_correct = "O" if label_idx == pred_idx else "X"
                
                if i < 10: 
                    print(f"| {fname[0]:<30} | {FIXED_CLASSES[label_idx]:^5} | {FIXED_CLASSES[pred_idx]:^5} | {is_correct:^5} |")

                if label_idx == pred_idx: correct += 1
                total += 1
        
        acc = correct / total * 100
        print(f"\n>>> [{filename_only}] 샘플 정확도: {acc:.2f}%")

if __name__ == "__main__":
    evaluate()
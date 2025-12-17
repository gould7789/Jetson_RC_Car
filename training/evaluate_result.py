import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
from PIL import Image

# ===============================================================
# [1] 설정 (경로 확인)
# ===============================================================
MODEL_PATH = "../models/pilotnet_steering_20251202_200510_best.pth" 
TEST_DATA_DIR = "../data-collector/dataset"
CLASSES = [10, 40, 70, 90, 110, 140, 170]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 때 사용했던 정확한 사이즈여야 합니다. (27% 나왔던 설정)
INPUT_HEIGHT = 66
INPUT_WIDTH = 200
# ===============================================================

# [2] 모델 구조
class RCNet(nn.Module):
    def __init__(self, num_classes=7):
        super(RCNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
            flat_size = self.features(dummy).view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# [3] 데이터셋 (다시 RGB로 복귀 + 정규화 유지)
class EvaluationDataset(Dataset):
    def __init__(self, img_dir, classes):
        self.img_paths = glob.glob(os.path.join(img_dir, "*"))
        self.classes = classes
        self.class_to_idx = {angle: i for i, angle in enumerate(classes)}
        
        # 27% 나왔던 설정으로 복귀하되, 정규화는 켭니다.
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.valid_data = [] 
        for path in self.img_paths:
            filename = os.path.basename(path)
            numbers = re.findall(r'\d+', filename)
            for num_str in numbers:
                angle = int(num_str)
                if angle in self.classes:
                    self.valid_data.append((path, self.class_to_idx[angle], angle)) # 실제 각도도 저장
                    break

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        path, label_idx, real_angle = self.valid_data[idx]
        try:
            image = Image.open(path).convert("RGB") # RGB가 맞았습니다 (BGR은 8% 나옴)
            image = self.transform(image)
            return image, label_idx, real_angle, path
        except:
            return torch.zeros(3, INPUT_HEIGHT, INPUT_WIDTH), label_idx, 0, path

# [4] 실행 및 진단
def run_evaluation():
    dataset = EvaluationDataset(TEST_DATA_DIR, CLASSES)
    loader = DataLoader(dataset, batch_size=32, shuffle=True) # 랜덤으로 섞어서 확인

    print(f"🤖 모델 로드 중...")
    model = RCNet(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- [진단] 실제 예측 눈으로 확인하기 ---
    print("\n" + "="*50)
    print("👀 [진단 모드] 실제 이미지 5개 예측 확인")
    print("="*50)
    
    with torch.no_grad():
        # 딱 1개 배치만 뽑아서 확인
        images, labels, real_angles, paths = next(iter(loader))
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # 5개만 출력
        for i in range(5):
            true_angle = real_angles[i].item()
            pred_idx = preds[i].item()
            pred_angle = CLASSES[pred_idx]
            
            result_str = "✅ 정답" if true_angle == pred_angle else "❌ 오답"
            print(f"파일: ...{os.path.basename(paths[i])[-15:]}")
            print(f"   -> 실제값: {true_angle:>3}°  vs  예측값: {pred_angle:>3}°  [{result_str}]")
            print("-" * 30)
    # -------------------------------------

    # 전체 평가 진행
    print("\n🚀 전체 데이터셋 평가 진행 중...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _, _ in DataLoader(dataset, batch_size=64, shuffle=False):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 결과 리포트
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n🏆 최종 Accuracy: {acc*100:.2f}%")
    print(f"🏆 최종 F1 Score: {f1:.4f}")
    
    # 혼동 행렬 저장
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    target_names = [f"{c}°" for c in CLASSES]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
    plt.savefig("diagnosis_result.png")
    print("💾 diagnosis_result.png 저장 완료")

if __name__ == "__main__":
    run_evaluation()
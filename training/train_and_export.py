import os
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# ==============================================================================
# [1] 설정 (사용자 환경)
# ==============================================================================
DATA_ROOT = "../data-collector/dataset"   # 데이터셋 경로
CLASSES = [10, 40, 70, 90, 110, 140, 170] # 클래스
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★ 변경점 1: 에폭 50 + 얼리 스탑핑 Patience 5 (더 엄격하게)
EPOCHS = 50           
PATIENCE = 5          # 5번 연속 성능 향상 없으면 바로 중단

# 전처리 설정 (도로만 보기 위해 Crop 유지)
INPUT_WIDTH = 200
INPUT_HEIGHT = 66
CROP_TOP_RATIO = 0.4  # 상단 40% 크롭
# ==============================================================================

# [2] 모델 구조 (PilotNet)
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
            self.flat_size = self.features(dummy).view(1, -1).size(1)
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# [3] 데이터셋 (올바른 전처리 적용)
class RCDataset(Dataset):
    def __init__(self, img_dir, classes):
        self.img_paths = glob.glob(os.path.join(img_dir, "*"))
        self.classes = classes
        self.class_to_idx = {angle: i for i, angle in enumerate(classes)}
        self.data = []

        print(f"📂 데이터 로드 중... (Crop Top {CROP_TOP_RATIO*100}%)")
        for path in self.img_paths:
            filename = os.path.basename(path)
            numbers = re.findall(r'\d+', filename)
            for num_str in numbers:
                angle = int(num_str)
                if angle in self.classes:
                    self.data.append((path, self.class_to_idx[angle]))
                    break
        print(f"✅ 총 {len(self.data)}개 로드 완료.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = cv2.imread(path)
        if image is None:
            return torch.zeros(3, INPUT_HEIGHT, INPUT_WIDTH), label
        
        h, w, _ = image.shape
        crop_y1 = int(h * CROP_TOP_RATIO)
        image = image[crop_y1:, :, :]
        image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), label

# [4] 메인 실행 함수
def main():
    # 데이터셋 준비
    full_dataset = RCDataset(DATA_ROOT, CLASSES)
    if len(full_dataset) == 0:
        print("❌ 데이터를 찾을 수 없습니다.")
        return

    # ★ 변경점 2: 데이터셋 분할 (학습 7 : 검증 2 : 테스트 1)
    total_len = len(full_dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.2)
    test_len = total_len - train_len - val_len # 남은거 다 테스트로 (약 10%)

    print(f"📊 데이터 분할: 학습({train_len}) / 검증({val_len}) / 테스트({test_len})")
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # 최종 평가용

    model = RCNet(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🚀 학습 시작 (Max Epochs: {EPOCHS}, Patience: {PATIENCE})...")
    print("="*60)

    best_val_loss = float('inf')
    patience_counter = 0
    best_acc_log = 0.0

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        # 1. Train (학습용 데이터 70%)
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # 2. Validation (검증용 데이터 20% -> Early Stopping 판단용)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_acc_log = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_final.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("\n⛔ Early Stopping! (5회 연속 성능 향상 없음)")
                print(f"   최고 기록: Val Loss {best_val_loss:.4f}, Val Acc {best_acc_log:.2f}%")
                break

    print("="*60)
    print(f"✅ 학습 종료. 최종 성능 평가를 진행합니다.")

    # --------------------------------------------------------
    # [5] 최종 평가 (테스트용 데이터 10% -> 보고서용)
    # --------------------------------------------------------
    # 학습된 최고의 모델 로드
    model.load_state_dict(torch.load("best_model_final.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    print("🧪 테스트 데이터셋(10%)에 대한 최종 검증 중...")
    with torch.no_grad():
        for imgs, labels in test_loader: # ★ 중요: test_loader 사용
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 리포트 출력
    target_names = [f"{c}°" for c in CLASSES]
    print("\n" + classification_report(all_labels, all_preds, target_names=target_names))
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    acc_final = accuracy_score(all_labels, all_preds)
    print(f"🏆 [최종 성적표] F1 Score: {f1:.4f}")
    print(f"🏆 [최종 성적표] Accuracy: {acc_final*100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Final Test Result (F1: {f1:.2f})")
    plt.savefig("final_test_result.png")
    print("💾 최종 결과 그래프: final_test_result.png")

    # --------------------------------------------------------
    # [6] ONNX 변환
    # --------------------------------------------------------
    print("\n📦 ONNX 변환 중...")
    dummy_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH).to(DEVICE)
    torch.onnx.export(
        model, dummy_input, "model.onnx",
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=['input'], output_names=['output']
    )
    print("✅ 모든 작업 완료!")

if __name__ == "__main__":
    main()
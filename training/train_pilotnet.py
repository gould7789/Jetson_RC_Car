import sys
import os

# [1] 경로 지정
current_dir = os.path.dirname(os.path.abspath(__file__)) # training 폴더
root_dir = os.path.dirname(current_dir)                  # 프로젝트 전체 폴더

sys.path.append(root_dir)    # preprocessor 찾기용
sys.path.append(current_dir) # 같은 폴더 파일(RCDataset 등) 찾기용

# [2] 라이브러리 및 모듈 가져오기
import time
import torch
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from torch import nn, optim

# ★ 경로 수정됨 (training. 을 뺌)
from RCDataset import RCDataset
from training.model import PilotNet
from preprocessor.RCPreprocessor import RCPreprocessor
from preprocessor.RCAugmentor import RCAugmentor

torch.backends.cudnn.benchmark = True

# ----------------------------------------------------
# [3] Early Stopping 클래스
# ----------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ----------------------------------------------------
# [4] 학습 메인 함수
# ----------------------------------------------------
def train():
    # --- 설정 ---
    csv_filename = "data_labels.csv"
    
    # ★ 하이픈(-) 경로 수정 적용
    dataset_root = os.path.join(root_dir, "data-collector", "dataset")
    
    num_epochs = 35
    batch_size = 128
    learning_rate = 1e-3
    patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # --- 데이터 로드 ---
    preproc = RCPreprocessor(out_size=(200, 66), crop_top_ratio=0.4)
    augment = RCAugmentor()

    # 경로 에러 방지를 위해 절대 경로(dataset_root) 사용
    train_dataset = RCDataset(csv_filename, dataset_root, preproc, augment, split="train")
    val_dataset   = RCDataset(csv_filename, dataset_root, preproc, split="val")
    test_dataset  = RCDataset(csv_filename, dataset_root, preproc, split="test")

    print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    num_classes = len(train_dataset.angles)
    print(f"[INFO] Classes: {num_classes} -> {train_dataset.angles}")

    num_workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- 모델 준비 ---
    model = PilotNet(num_classes=num_classes, input_shape=(3, 66, 200)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 저장 폴더 (프로젝트 루트/models)
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    temp_checkpoint_path = os.path.join(models_dir, "best_checkpoint.pth")
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=temp_checkpoint_path)

    # 그래프 기록용
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # --- 학습 루프 ---
    print(f"\n=== Start Training (Epochs: {num_epochs}) ===")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # 1. Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        ep_train_loss = train_loss / train_total
        ep_train_acc  = train_correct / train_total * 100.0

        # 2. Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        ep_val_loss = val_loss / val_total
        ep_val_acc  = val_correct / val_total * 100.0

        # 기록
        history['train_loss'].append(ep_train_loss)
        history['val_loss'].append(ep_val_loss)
        history['train_acc'].append(ep_train_acc)
        history['val_acc'].append(ep_val_acc)

        print(f"[Epoch {epoch:02d}] Loss: {ep_train_loss:.4f} / {ep_val_loss:.4f} | Acc: {ep_train_acc:.2f}% / {ep_val_acc:.2f}%")

        # 3. Early Stopping
        early_stopping(ep_val_loss, model)
        if early_stopping.early_stop:
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    print(f"Total Time: {time.time() - start_time:.2f}s")

    # --- 최종 저장 ---
    print("\n[INFO] Loading best model...")
    model.load_state_dict(torch.load(temp_checkpoint_path))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(models_dir, f"pilotnet_steering_{timestamp}_best.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Best Model Saved: {final_path}")

    # --- Test ---
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    final_acc = test_correct / test_total * 100.0 if test_total > 0 else 0.0
    print(f"★ Final Test Accuracy: {final_acc:.2f}%")

    # --- 그래프 그리기 ---
    plot_path = os.path.join(models_dir, f"training_graph_{timestamp}.png")
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'b-', label='Train')
    plt.plot(epochs_range, history['val_loss'], 'r--', label='Val')
    plt.title('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], 'b-', label='Train')
    plt.plot(epochs_range, history['val_acc'], 'r--', label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[INFO] Graph Saved: {plot_path}")
    # plt.show() # 에러 방지를 위해 주석 처리 (저장만 해도 충분)

if __name__ == "__main__":
    train()
import torch
import torch.onnx
import os
import sys

# [1] 경로 설정 (training 폴더 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# [2] 학습 때 사용한 모델 클래스 가져오기
# (train_pilotnet.py에서 썼던 것과 똑같은 걸 가져와야 합니다!)
try:
    from model import PilotNet
    print("[INFO] PilotNet 모듈 로딩 성공")
except ImportError:
    print("[ERROR] 'preprocessor/model.py'를 찾을 수 없습니다. 경로를 확인하세요.")
    sys.exit()

def export_onnx():
    # ==========================================
    # ★ 수정할 부분: 변환할 .pth 파일 이름
    # ==========================================
    pth_filename = "../models/pilotnet_steering_20251213_224453_best.pth"  # 여기에 실제 파일명을 적으세요
    
    input_pth_path = os.path.join(root_dir, "models", pth_filename)
    output_onnx_path = os.path.join(root_dir, "models", pth_filename.replace(".pth", ".onnx"))

    # 1. 모델 초기화 (구조 생성)
    # 학습할 때 num_classes=7, input_shape=(3, 66, 200)을 썼다고 가정합니다.
    device = torch.device("cpu") # 변환은 CPU에서 해도 충분합니다.
    model = PilotNet(num_classes=7, input_shape=(3, 66, 200)).to(device)

    # 2. 가중치(.pth) 로드
    if not os.path.exists(input_pth_path):
        print(f"[ERROR] 파일이 없습니다: {input_pth_path}")
        return

    print(f"[INFO] .pth 모델 로드 중... ({input_pth_path})")
    # map_location='cpu'는 GPU에서 학습한 모델을 CPU로 불러올 때 필수입니다.
    model.load_state_dict(torch.load(input_pth_path, map_location=device))
    model.eval() # 평가 모드로 전환 (Dropout 제거 등)

    # 3. 더미 데이터 생성 (모델이 입력받을 모양과 똑같은 가짜 데이터)
    # (배치크기 1, 채널 3, 높이 66, 너비 200)
    dummy_input = torch.randn(1, 3, 66, 200, device=device)

    # 4. ONNX로 변환 (Export)
    print(f"[INFO] ONNX 변환 시작... -> {output_onnx_path}")
    torch.onnx.export(
        model,                      # 실행할 모델
        dummy_input,                # 모델 입력값 (모양 체크용)
        output_onnx_path,           # 저장될 파일 경로
        export_params=True,         # 가중치 포함 여부
        opset_version=11,           # ★ 젯슨 나노 호환성을 위해 11 버전 추천
        do_constant_folding=True,   # 최적화 (상수 폴딩)
        input_names=['input_image'],   # 입력 노드 이름 (나중에 젯슨에서 중요)
        output_names=['steering_angle'] # 출력 노드 이름
    )

    print("="*50)
    print(f"✅ 변환 완료! 파일 위치:\n{output_onnx_path}")
    print("="*50)
    print("이제 이 .onnx 파일을 젯슨 나노로 옮겨서 변환하세요.")

if __name__ == "__main__":
    export_onnx()
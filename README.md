# 🚗 딥러닝 기반 자율주행 자동차 (Capstone Design)

<!-- 뱃지: 사용한 기술 스택을 보여줍니다 -->
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white"/> <img src="https://img.shields.io/badge/NVIDIA-76B900?style=flat-square&logo=NVIDIA&logoColor=white"/> <img src="https://img.shields.io/badge/Jetson_Nano-76B900?style=flat-square&logo=NVIDIA&logoColor=white"/>

## 📖 프로젝트 개요
- **과목명**: 캡스톤디자인
- **진행 기간**: 2025.08.25 ~ 2025.12.14 (16주)
- **개발 목표**: 
  - NVIDIA Jetson Nano와 딥러닝(CNN)을 활용한 End-to-End 자율주행 구현
  - 차선 인식(Lane Keeping) 및 신호등 인식에 따른 주행 제어
- **성과**: 
  - ResNet-18 기반 모델 학습 (정확도 **82.77%** 달성)
  - 데이터 불균형 및 과적합 문제를 해결하여 안정적인 주행 성공

---

## 🛠 1. 하드웨어 구성

<!-- 요청하신 도식과 실제 사진을 나란히 배치하는 테이블 구조입니다 -->
<table>
  <tr>
    <td align="center" width="50%">
      <b>📌 시스템 구상 도식</b>
    </td>
    <td align="center" width="50%">
      <b>🚗 실제 제작 결과물</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <!-- 기존에 가지고 계신 도식 이미지 링크 -->
      <img src="https://github.com/user-attachments/assets/18e34af2-e041-413a-9ece-fe54d7c96703" width="90%" />
    </td>
    <td align="center">
      <!-- [사진 넣는 곳 1] 실제 자동차 사진 파일 경로를 아래 src="" 안에 넣어주세요 -->
      <img src="https://github.com/user-attachments/assets/30b3b9e1-fd9a-4ee8-bcca-1a03c95a8b0b" alt="실제 자율주행차 사진" width="90%" />
    </td>
  </tr>
</table>

### 🔧 주요 부품 사양
| 부품명 | 역할 | 비고 |
|:---:|:---|:---|
| **Jetson Nano** | 메인 임베디드 보드 | AI 모델 연산 및 제어 |
| **CSI Camera** | 영상 데이터 수집 | 전방 라인 및 신호 인식 |
| **DC Motor** | 차량 구동 (후륜) | 속도 제어 |
| **Servo Motor** | 차량 조향 (전륜) | 방향 제어 |

---

## 🧠 2. 학습 프로세스 및 기술적 접근

### 2-1. 데이터 수집 및 전처리
- **수집 방법**: 조이스틱을 이용한 수동 주행 영상 & 조향각(Label) 동시 수집
- **데이터셋 비율**: `Train(70%)` : `Val(20%)` : `Test(10%)`
  > **전략**: Test Set(10%)은 학습에 전혀 관여하지 않도록 분리하여 객관적인 주행 성능 평가 지표로 활용함.

### 2-2. 모델 학습 (Training)
- **Network**: **ResNet-18** (Pre-trained Model 활용)
- **Hyperparameter Tuning**:
  - `Epochs`: 초기 20회 → **50회**로 상향 (과소적합 방지)
  - `Early Stopping`: 과적합(Overfitting) 방지를 위해 Validation Loss가 개선되지 않으면 학습 조기 종료

---

## 📉 3. 트러블 슈팅 (시행착오 과정)

프로젝트 진행 중 발생한 문제와 이를 해결한 과정입니다.

### ⚠️ [1차 시도] 데이터 불균형 문제
- **현상**: 직진과 좌회전은 잘 수행하나, **우회전 구간에서 경로를 이탈**함.
- **원인**: 트랙 구조상 우회전 데이터가 턱없이 부족했음 (Angle Distribution 불균형).

### ⚠️ [2차 시도] 잘못된 오버샘플링(Data Augmentation)
- **시도**: 부족한 데이터를 늘리기 위해 이미지를 기계적으로 좌우 반전(Flip) 시킴.
- **결과**: 오히려 기존에 잘 되던 좌회전 성능까지 하락.
- **원인**: 배경의 조명, 그림자 위치 등이 반전되면서 현실 세계와 다른 **노이즈(Noise)**로 작용함.

### ✅ [3차 시도] 최적화 성공
- **해결**: 인위적인 증강을 줄이고, 부족한 각도(Class)의 데이터를 추가 주행하여 직접 확보.
- **결과**: 모든 클래스(각도)의 데이터 균형을 맞춤.
  - **Loss**: 0.1 이하로 안정적 수렴
  - **Test Accuracy**: **82.77%** 달성
  - **주행**: 라인 이탈 없는 부드러운 코너링 구현 성공

---

## 📊 4. 최종 성능 평가

<!-- [사진 넣는 곳 3] 보고서 7번 항목의 '성능 평가지표 및 분석' 혼동 행렬(Confusion Matrix) 이미지 -->
<img src="https://github.com/user-attachments/assets/cd291351-b573-4ead-a6f7-5f4d8303adc3" width="80%" />

| Metric | Score | 의미 |
|:---:|:---:|---|
| **Accuracy** | **82.77%** | 전체 상황 중 약 83%에서 올바른 조향각 판단 |
| **Precision** | 83.00% | 모델이 정답이라 예측한 것 중 실제 정답 비율 |
| **Recall** | 83.00% | 실제 데이터를 놓치지 않고 찾아낸 확률 |
| **F1-Score** | 82.68% | 정밀도와 재현율의 조화 평균 |

> 특히 사고 위험이 높은 **급커브(170도)** 구간에서 높은 인식률을 보여 성공적인 Lane Keeping이 가능함을 입증함.

---

## 📅 5. 프로젝트 타임라인

| 주차 | 주요 활동 | 비고 |
|:---:|---|---|
| 1~2주 | 프로젝트 기획 및 하드웨어 구상 | 계획서 작성 |
| 3~5주 | 하드웨어 제작 및 조립, RC카 구동계 테스트 | |
| 6~8주 | OpenCV 영상 처리 및 데이터 수집 환경 구축 | 중간 발표 |
| 9~11주 | 데이터셋 수집 및 1, 2차 모델 학습 | 시행착오(불균형) |
| 12~15주 | 모델 최적화(3차 학습) 및 자율주행 테스트 | 최종 성능 82.77% |
| 16주 | 최종 프로젝트 발표 및 시연 | |

---

## 👤 작성자 (Author)

- **이름**: 이현우
- **소속**: 영진전문대학교 글로벌시스템융합과
- **역할**: 
  - 자율주행 하드웨어 제작 및 제어
  - PyTorch 기반 CNN 모델 설계 및 학습
  - 데이터 불균형 문제 분석 및 해결

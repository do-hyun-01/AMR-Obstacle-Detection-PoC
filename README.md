# 제조 현장 AMR의 안전 주행을 위한 딥러닝 기반 실시간 동적 장애물 탐지 PoC 프로젝트 : Deep Learning-based Real-time Dynamic Obstacle Detection for AMR Safe Driving

본 프로젝트는 제조 현장 내 자율 주행 로봇(AMR)의 안전 주행을 위한 YOLO26 기반 실시간 장애물 감지 시스템 개발을 목표로 합니다. G-RISE PoC(Proof of Concept) 연구의 일환으로 진행되었으며, 현장 데이터 적응(Domain Adaptation)을 통해 낮은 카메라 각도에서의 인식 성능을 극대화했습니다.

## 📅 1. 프로젝트 일정
- **수행 기간**: 2026.01.15 ~ 2026.01.26

## 💻 2. 기술 스택 및 개발 환경

### 🛠 Hardware & OS
* **Cloud**: Naver Cloud Platform (NCP)
* **GPU**: NVIDIA Tesla T4 (16GB VRAM) x 1
* **OS**: Windows Server 2019

### 📚 Frameworks & Libraries
* **Language**: ![Python](https://img.shields.io/badge/Python-3.12.0-3776AB?logo=python&logoColor=white)
* **Deep Learning**: ![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white) ![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.5-007ACC?logo=yolo&logoColor=white)
* **Computer Vision**: ![OpenCV](https://img.shields.io/badge/OpenCV-Latest-5C3EE8?logo=opencv&logoColor=white)

### ⚙️ Environment Details
* **CUDA**: 12.2 (Driver Version: 537.13)
* **Model**: YOLO26s (Small architecture optimized for AMR)

## 📊 3. 데이터셋 (AI-Hub)
- 121.로봇 관점 주행 영상 데이터 - S(특수상황) - I(산업시설)의 28,229장의 이미지
- CVAT을 통해 직접 라벨링한 1,264장의 이미지 추가

## ⚡️ 4. Key Improvements (baseline to fine-tuning)

**Problem: Domain Gap & Misclassification**

- **Low-angle Viewpoint**: 테스트 영상의 낮은 카메라 각도에 적응하지 못해 화면 전체를 `fixed_object`로 오인하는 문제 발생.
- **Detection Failure**: 주행 중인 `moving_object`에 대한 인식률 저하.

**Solution: Data-Centric Domain Adaptation**

- **Field Data Adaptation**: 테스트 영상에서 추출한 **1,264세트의 정밀 라벨링 데이터**를 활용해 도메인 적응 수행.
- **Label Integration**: 로봇 본체와 상단 적재물(Rack)을 `moving_object`로 통합 라벨링하여 **회피 경로 계산의 안정성 확보**.

## 📂 5. Directory Structure

```text
AMR-OBSTACLE-DETECTION-POC/
├── config/
│   └── data.yaml             # 클래스 정의 및 데이터 경로
├── models/
│   └── weights/              # 최종 베스트/라스트 가중치 (best.pt / last.pt)
├── results/                  # 학습 로그 및 성능 분석 지표
│   ├── yolo26_final          # 베이스 모델 학습 결과 (100 Epochs)
│   └── yolo26_fine_tuning    # 파인튜닝 적응 결과 (50 Epochs)
├── scripts/                  # 데이터 전처리 및 학습 스크립트
│   ├── Additional_datasets.py # 추가 라벨링 데이터 분류 및 이동
│   ├── Fine-tuning.py         # YOLO26 파인튜닝 실행 코드
│   ├── Preprocessing.py       # 데이터셋 구축 및 Train/Val 분할
│   ├── Resize.py              # 이미지 640px 리사이징
│   └── YOLO_label_merge.py    # 로봇과 적재물을 moving_object로 통합
├── .gitignore                # Git 업로드 제외 설정
├── README.md                 # 프로젝트 상세 리포트
└── requirements.txt          # 개발 환경 의존성 패키지 목록
```

## 📈 6. Performance Analysis

| Metric | Baseline Model | **Fine-tuned Model** | Comparison |
| --- | --- | --- | --- |
| **Box Loss** | 0.599 | **0.528** | **약 11.8% 정밀도 향상** |
| **Class Loss** | 0.325 | **0.279** | **약 14.1% 분류 성능 향상** |
| **mAP50-95** | 0.671 | **0.667** | **전체 데이터셋 안정성 유지** |

**핵심 결과 분석**
- Loss 하락: 40 에포크 이후 손실값이 급격히 하락하며 현장 환경의 시각적 특징을 정확히 포착함.

- 객체 분별력: moving_object의 Precision이 0.835를 기록하며, 로봇 주행 중 오검출로 인한 급정거 문제를 획기적으로 개선함.

## 7. 탐지 시연
(결과 GIF 또는 이미지를 첨부 예정)

# 제조 현장 AMR의 안전 주행을 위한 딥러닝 기반 실시간 동적 장애물 탐지 PoC 프로젝트 : Deep Learning-based Real-time Dynamic Obstacle Detection for AMR Safe Driving

본 프로젝트는 물류창고 및 제조 현장에서 발생할 수 있는 충돌 사고를 방지하기 위해, YOLOv8 모델을 활용하여 사람, 지게차, 파렛트_랙 등의 장애물을 고정밀도로 탐지하는 것을 목표로 합니다. 특히 v3 모델에서 발견된 오탐지(False Positive) 및 미탐지(False Negative) 문제를 데이터 공학적 관점에서 해결한 v4 개선 과정을 중점적으로 다룹니다.

## 📅 1. 프로젝트 일정
- **수행 기간**: 2026.01.15 ~ 2026.01.26

## 💻 2. 기술 스택 및 개발 환경
- **Environment**: Naver Cloud Platform (Tesla T4 GPU 1EA, 16GB VRAM)
- **OS**: Windows Server 2019
- **Framework**: PyTorch 2.0+, Ultralytics (YOLOv8)
- **Tools**: OpenCV, Python 3.12

## 📊 3. 데이터셋 (AI-Hub)
- 102번 건설기계 무인 운행 데이터
- 107번 로봇 관점 주행 영상_고도화_소셜 내비게이션 로봇 주행
- 121번 물류창고 내 작업 안전 데이터

## ⚡️ 4. Key Improvements (v1,2 to v3 to v4)
v1,2 학습 모델에서 발생한 장애물 클래스 인식 누락 및 라벨링 오류, 데이터셋 통합 스크립트의 클래스 매핑 오류 및 라벨 오염 수정 문제를 해결하였고, v3 학습 모델에서 발생한 지게차 오탐지와 파렛트 랙 미탐지 문제를 해결하기 위해 세 가지 전략을 적용했습니다.

### Strategy 1: Hard Negative Mining (Background Images)

- **문제**: 창고 내 기둥이나 배전함 등 금속 재질의 수직 구조물을 지게차로 오인함.
- **해결**: AI-Hub 102번(건설기계 무인운행) 데이터에서 **지게차와 질감이 유사한 중장비(덤프트럭, 포크레인 등) 사진 700장**을 추출하여 라벨 없는 **배경 이미지(Background Images)**로 투입.
- **결과**: 지게차 Precision(정밀도)이 **0.987**로 상승하며 오탐지 문제 해결.

### Strategy 2: Data Augmentation 강화

- **문제**: 밀집된 환경이나 원거리에 있는 파렛트 랙을 인식하지 못함.
- **해결**:
    - `mosaic=1.0`: 4장의 이미지를 합쳐 객체 밀도를 높임.
    - `mixup=0.15`: 이미지 중첩을 통해 객체 경계 학습 강화.
    - `scale=0.9`: 다양한 거리감 학습 유도.
- **결과**: 파렛트 랙 Recall(재현율) **0.983** 달성.

### Strategy 3: Hyperparameter Tuning

- **Loss Weight**: 분류 손실 가중치(`cls`)를 **2.0**으로 상향하여 클래스 간 변별력 강화.
- **HSV-V**: 명도 증강(`hsv_v=0.4`)을 통해 조명 변화에 강건한 모델 구축.

## ✨ 5. Performance Evaluation
| **Metric** | **v3 (Baseline)** | **v4 (Improved)** | **Change** |
| --- | --- | --- | --- |
| **mAP50 (All)** | 0.865 | **0.989** | **+12.4%** |
| **mAP50-95** | - | **0.890** | **-** |
| **Forklift Precision** | 0.784 | **0.987** | **+20.3%** |
| **Pallet_Rack Recall** | 0.821 | **0.983** | **+16.2%** |


## 📈 6. 연구 결과 (PoC 검증)

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/d452db68-2029-4586-b609-7d1473ceba53" />

| **Version** | **Description** | **Target** | **Key Strategy** | **Result (mAP50)** |
| --- | --- | --- | --- | --- |
| v1-v2 | Base Training | 기초 탐지 | 데이터 통합 | 0.820 |
| v3 | Full Training | 실무 검증 | 에포크 확장 및 기본 학습 | 0.865 |
| v4 | Final Fix | 성능 최적화 | 배경 이미지 보강 + 증강 강화 | 0.989 |


## 7. 탐지 시연
(결과 GIF 또는 이미지를 첨부 예정)

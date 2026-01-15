# 제조 현장 AMR 안전 주행을 위한 실시간 장애물 탐지 PoC

제조 현장에서의 안전 사고 예방을 위해 자율주행 로봇(AMR) 시점의 데이터를 활용한 실시간 동적 장애물 탐지 모델 연구입니다.

## 📅 프로젝트 일정
- **수행 기간**: 2026.01.15 ~ 2026.01.26 (PoC 완료 예정)

## 💻 개발 환경
- **Server**: Naver Cloud Platform (Tesla T4 GPU 1EA, 16GB VRAM)
- **OS**: Windows Server 2019
- **Framework**: PyTorch, Ultralytics (YOLOv8)

## 📊 데이터셋 (AI-Hub)
- 물류창고 내 작업 안전 데이터
- 로봇 관점 주행 영상 (소셜 내비게이션)
- **추출 클래스**: 작업자(Person), 지게차(Forklift), 박스(Box)

## 🚀 연구 결과 목표 (PoC 검증)
| Metric | Target | Result |
| :--- | :--- | :--- |
| **mAP@0.5** | 0.5 이상 | (업데이트 예정) |
| **Inference Speed** | 30 FPS 이상 | (업데이트 예정) |

## 탐지 시연
(결과 GIF 또는 이미지를 첨부 예정)

## **데이터 전처리 및 통합 과정 (Data Preprocessing)**

본 문서는 AMR(Autonomous Mobile Robot)의 안전 주행 PoC를 위해 수행한 데이터셋 통합 및 전처리 과정을 기록합니다.

### **1. 데이터셋 개요**

- **Dataset 121.물류창고 내 작업 안전 데이터**: 물류 창고 및 제조 현장의 지게차, 파렛트 중심 데이터.
- **Dataset 107.로봇 관점 주행 영상_고도화_소셜 내비게이션 로봇 주행**: 로봇 관점(Ego-vision)에서 촬영된 보행자 및 실내 장애물 데이터.

### **2. 클래스 정의 및 매핑 (Class Mapping)**

두 데이터셋의 클래스 체계를 통합하기 위해 다음과 같이 매핑을 수행하였습니다.

| 클래스명 | **Dataset 121 ID** | **Dataset 107 ID** | **통합 ID (YOLO)** |
| --- | --- | --- | --- |
| **Person** | 0 | 14 | 0 |
| **Forklift** | 1 | - | 1 |
| **Pallet_Rack** | 2 | - | 2 |
| **Box** | 3 | 10 | 3 |
- Dataset 107의 `person(14)`을 통합 ID `0`으로 강제 매핑하여 학습의 일관성을 확보하였습니다.

### **3. 좌표 체계 변환 및 정규화 (BBox Normalization)**

Dataset 107은 `[x1, y1, x2, y2]` 형식을 사용하므로, 이를 YOLOv8 학습에 적합한 `[x_center, y_center, width, height]` 정규화 형식으로 변환하였습니다.

**변환 공식**

1. **Width/Height 계산**:
    - $w = x_2 - x_1$
    - $h = y_2 - y_1$
2. **YOLO 포맷 정규화**:
    - $x_{center} = \frac{x_1 + \frac{w}{2}}{w_{orig}}$
    - $y_{center} = \frac{y_1 + \frac{h}{2}}{h_{orig}}$
    - $w_{norm} = \frac{w}{w_{orig}}, \quad h_{norm} = \frac{h}{h_{orig}}$

### **4. 데이터 샘플링 및 분할 (Sampling & Split)**

- **시간적 샘플링**: 107번 데이터의 연속 프레임 간 중복성을 줄이기 위해 10프레임당 1장을 추출하는 샘플링(1/10)을 적용하였습니다.
- **데이터 분할**: 전체 통합 데이터셋을 학습용(Train) 80%, 검증용(Val) 20% 비율로 무작위 분할하였습니다.

### **5. 주요 이슈 및 해결 (Troubleshooting)**

- **인코딩 오류**: Windows 터미널의 한글 깨짐 현상을 `chcp 65001` 설정을 통해 해결하였습니다.
- **확장자 대소문자**: `.jpg`와 `.JPG` 확장자가 혼용된 데이터를 모두 처리할 수 있도록 소스 코드를 최적화하였습니다.

import json
import os
import cv2
import numpy as np
from tqdm import tqdm

# === 1. 경로 설정 ===
BASE_LABEL_DIR = r"D:\107.로봇 관점 주행 영상_고도화_소셜 내비게이션 로봇 주행\3.개방데이터\1.데이터\Training\02.라벨링데이터\TL"
BASE_IMAGE_DIR = r"D:\107.로봇 관점 주행 영상_고도화_소셜 내비게이션 로봇 주행\3.개방데이터\1.데이터\Training\01.원천데이터\TS"
SAVE_DIR = r"C:\AMR_Dataset_Final"

# 107번 데이터를 121번 클래스 체계로 매핑
# 107번에 지게차가 없다면 Person과 Box(Cargo) 위주로 추출.
class_mapping = {
    "Person": 0, "Pedestrian": 0,  # 사람 -> 0번
    "Box": 2, "Cargo": 2,          # 박스/적재물 -> 2번
    "Forklift": 1                  # 혹시 있을 경우 -> 1번
}

SAMPLING_RATE = 10 # 10프레임당 1개 (PoC 속도 향상)
IMG_SIZE = 640

# 폴더 생성
os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "labels"), exist_ok=True)

json_files = [f for f in os.listdir(BASE_LABEL_DIR) if f.endswith('.json')]

for i, j_file in enumerate(tqdm(json_files, desc="107번 데이터 처리 중")):
    if i % SAMPLING_RATE != 0: continue
    
    # 중복 확인: 이미 있으면 건너뜀
    if os.path.exists(os.path.join(SAVE_DIR, "labels", j_file.replace('.json', '.txt'))):
        continue

    with open(os.path.join(BASE_LABEL_DIR, j_file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 107번 데이터의 시각 정보/해상도 추출 (구조에 따라 수정 필요할 수 있음)
    try:
        annotations = data['annotations'] # 107번 일반적 구조
        w_orig = data['metadata']['width']
        h_orig = data['metadata']['height']
        img_name = data['metadata']['file_name']
    except KeyError:
        continue # 구조가 다를 경우 건너뜀

    yolo_labels = []
    for ann in annotations:
        label = ann.get('category') or ann.get('class_id')
        if label in class_mapping:
            cid = class_mapping[label]
            # 좌표 변환 (x, y, w, h)
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / w_orig
            y_center = (y + h / 2) / h_orig
            yolo_labels.append(f"{cid} {x_center:.6f} {y_center:.6f} {(w/w_orig):.6f} {(h/h_orig):.6f}")

    if yolo_labels:
        img = cv2.imdecode(np.fromfile(os.path.join(BASE_IMAGE_DIR, img_name), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(os.path.join(SAVE_DIR, "images", img_name), cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            with open(os.path.join(SAVE_DIR, "labels", j_file.replace('.json', '.txt')), 'w') as f:
                f.write("\n".join(yolo_labels))
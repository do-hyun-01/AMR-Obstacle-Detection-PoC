import json
import os
import cv2
import numpy as np
from tqdm import tqdm

# === 1. 경로 설정 ===
BASE_LABEL_DIR = r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_07_지게차\작업상황(WS)"
BASE_IMAGE_DIR = r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_07_지게차\작업상황(WS)"
SAVE_DIR = r"C:\AMR_Dataset_Final"

class_mapping = {"WO-01": 0, "WO-04": 1, "SO-02": 2, "WO-03": 3}
SAMPLING_RATE = 10  # 10개 중 1개 샘플링
IMG_SIZE = 640      # 리사이징 크기

# 한글 경로 이미지를 읽기 위한 함수
def imread_korean(path):
    try:
        with open(path, "rb") as f:
            bytes = bytearray(f.read())
            array = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except:
        return None

# 폴더 생성
os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "labels"), exist_ok=True)

# 전처리 루프
json_files = [f for f in os.listdir(BASE_LABEL_DIR) if f.endswith('.json')]
selected_count = 0
skipped_count = 0

print(f"작업 시작... 대상 파일: {len(json_files)}개")

for i, j_file in enumerate(tqdm(json_files)):
    # 1. 샘플링 (10개당 1개)
    if i % SAMPLING_RATE != 0: continue

    # 2. 중복 체크
    label_save_path = os.path.join(SAVE_DIR, "labels", j_file.replace('.json', '.txt'))
    if os.path.exists(label_save_path):
        skipped_count += 1
        continue

    # 3. JSON 읽기
    with open(os.path.join(BASE_LABEL_DIR, j_file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 4. 필수 객체 포함 여부 확인 (사람 또는 지게차)
    annotations = data['Learning data info.']['annotation']
    has_target = any(ann['class_id'] in ["WO-01", "WO-04"] for ann in annotations)
    if not has_target: continue

    # 5. 이미지 읽기
    img_name = data['Source data Info.']['source_data_ID'] + ".jpg"
    img_path = os.path.join(BASE_IMAGE_DIR, img_name)
    
    if not os.path.exists(img_path): continue
    
    img = imread_korean(img_path)
    if img is None: continue

    # 6. YOLO 변환 및 저장
    h_orig, w_orig = data['Raw data Info.']['resolution']
    yolo_labels = []
    for ann in annotations:
        if ann['class_id'] in class_mapping:
            cid = class_mapping[ann['class_id']]
            x, y, w, h = ann['coord']
            
            x_center = (x + w / 2) / w_orig
            y_center = (y + h / 2) / h_orig
            w_norm = w / w_orig
            h_norm = h / h_orig
            yolo_labels.append(f"{cid} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    if yolo_labels:
        # 이미지 리사이징 및 저장
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(SAVE_DIR, "images", img_name), img_resized)
        
        # 라벨 저장
        with open(label_save_path, 'w') as f:
            f.write("\n".join(yolo_labels))
        selected_count += 1

print(f"\n--- 작업 결과 보고 ---")
print(f"새롭게 추가된 데이터: {selected_count}개")
print(f"이미 존재하여 건너뛴 데이터: {skipped_count}개")
print(f"최종 저장 경로: {SAVE_DIR}")
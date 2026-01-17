import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# === 1. 설정 및 경로 ===
SAVE_DIR = r"C:\AMR_Dataset_Final"
IMG_SIZE = 640
SAMPLING_RATE = 10 

# 클래스 매핑 (0:Person, 1:Forklift, 2:Pallet_Rack, 3:Box)
mapping_107 = {"Person": 0, "Pedestrian": 0, "Forklift": 1, "Box": 3, "Cargo": 3}
mapping_121 = {"WO-01": 0, "WO-04": 1, "SO-02": 2, "WO-03": 3}

# === 2. 처리 대상 폴더 리스트 (사용자 데이터셋 경로 완벽 반영) ===
tasks = [
    # 107번 데이터셋
    (r"D:\107.로봇 관점 주행 영상_고도화_소셜 내비게이션 로봇 주행\3.개방데이터\1.데이터\Training\02.라벨링데이터\TL",
     r"D:\107.로봇 관점 주행 영상_고도화_소셜 내비게이션 로봇 주행\3.개방데이터\1.데이터\Training\01.원천데이터\TS",
     "107_", mapping_107, "107"),
    
    # 121번 지게차 (FL)
    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_07_지게차\작업상황(WS)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_07_지게차\작업상황(WS)",
     "121_FL_WS_", mapping_121, "121"),
    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_07_지게차\불안전한 행동(UA)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_07_지게차\불안전한 행동(UA)",
     "121_FL_UA_", mapping_121, "121"),
    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_07_지게차\불안전한 상태(UC)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_07_지게차\불안전한 상태(UC)",
     "121_FL_UC_", mapping_121, "121"),

    # 121번 파렛트/랙 (RK)
    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_09_파렛트,렉\작업상황(WS)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_09_파렛트,렉\작업상황(WS)",
     "121_RK_WS_", mapping_121, "121"),
    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_09_파렛트,렉\불안전한 상태(UC)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_09_파렛트,렉\불안전한 상태(UC)",
     "121_RK_UC_", mapping_121, "121")
]

# 폴더 초기화
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "labels"), exist_ok=True)

def imread_korean(path):
    try:
        with open(path, "rb") as f:
            return cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    except: return None

def find_image_robust(img_dir, base_name):
    """확장자 대소문자 구분 없이 이미지 찾기"""
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG']:
        path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(path):
            return path
    return None

# === 3. 통합 전처리 루프 ===
total_saved = 0
for label_dir, img_dir, prefix, mapping, d_type in tasks:
    if not os.path.exists(label_dir):
        print(f"⚠️ 경로 없음 (건너뜀): {label_dir}")
        continue
    
    json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    print(f"\n🚀 처리 시작: {prefix} (대상: {len(json_files)}개)")
    
    for i, j_file in enumerate(tqdm(json_files)):
        if i % SAMPLING_RATE != 0: continue
        
        with open(os.path.join(label_dir, j_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        yolo_labels = []
        w_orig, h_orig, img_name = None, None, None
        
        try:
            if d_type == "107":
                annotations = data.get('annotations', [])
                # KeyError 방지를 위한 안전한 키 접근
                meta = data.get('metadata') or data.get('image') or {}
                w_orig = meta.get('width')
                h_orig = meta.get('height')
                img_name = meta.get('file_name')
                img_base = os.path.splitext(img_name)[0] if img_name else None
            else: # 121
                annotations = data['Learning data info.']['annotation']
                h_orig, w_orig = data['Raw data Info.']['resolution']
                img_base = data['Source data Info.']['source_data_ID']

            if not all([w_orig, h_orig, img_base]): continue

            for ann in annotations:
                cat = ann.get('category') or ann.get('class_id')
                if cat in mapping:
                    cid = mapping[cat]
                    x, y, w, h = ann.get('bbox') or ann.get('coord')
                    x_c, y_c = (x + w/2)/w_orig, (y + h/2)/h_orig
                    yolo_labels.append(f"{cid} {x_c:.6f} {y_c:.6f} {w/w_orig:.6f} {h/h_orig:.6f}")

            if yolo_labels:
                actual_path = find_image_robust(img_dir, img_base)
                if actual_path:
                    img = imread_korean(actual_path)
                    if img is not None:
                        ext = os.path.splitext(actual_path)[1]
                        save_name = f"{prefix}{img_base}{ext}"
                        cv2.imwrite(os.path.join(SAVE_DIR, "images", save_name), cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
                        with open(os.path.join(SAVE_DIR, "labels", prefix + j_file.replace('.json', '.txt')), 'w') as f:
                            f.write("\n".join(yolo_labels))
                        total_saved += 1
        except Exception as e:
            continue

print(f"\n✅ 최종 구축 완료! 총 {total_saved}개 데이터 저장 위치: {SAVE_DIR}")
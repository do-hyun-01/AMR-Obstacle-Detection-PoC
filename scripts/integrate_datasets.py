import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import glob

# === 1. 설정 및 경로 ===
SAVE_DIR = r"C:\AMR_Dataset_Final"
IMG_SIZE = 640

# 클래스 매핑 (0:Person, 1:Forklift, 2:Pallet_Rack, 3:Box)
mapping_107 = {"14": 0, "person": 0, "Person": 0}
mapping_121 = {
    "WO-01": 0,              # 작업자
    "SO-01": 1, "WO-04": 1,  # 지게차
    "SO-02": 2, "SO-13": 2,  # 파렛트 랙/파렛트
    "WO-03": 3, "WO-02": 3, "WO-05": 3 # 운반물/적재물/낙하물
}

# (라벨폴더, 이미지폴더, 접두어, 매핑사전, 데이터타입, 샘플링비율)
tasks = [
    (r"D:\107.로봇 관점 주행 영상_고도화_소셜 내비게이션 로봇 주행\3.개방데이터\1.데이터\Training\02.라벨링데이터\TL",
     r"D:\107.로봇 관점 주행 영상_고도화_소셜 내비게이션 로봇 주행\3.개방데이터\1.데이터\Training\01.원천데이터\TS",
     "107_", mapping_107, "107", 1), # 107번은 전수 사용
    
    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_07_지게차\작업상황(WS)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_07_지게차\작업상황(WS)",
     "121_FL_WS_", mapping_121, "121", 20), # 20장당 1장
    
    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_07_지게차\불안전한 상태(UC)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_07_지게차\불안전한 상태(UC)",
     "121_FL_UC_", mapping_121, "121", 20),

    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_09_파렛트,렉\작업상황(WS)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_09_파렛트,렉\작업상황(WS)",
     "121_RK_WS_", mapping_121, "121", 10),

    (r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\라벨링데이터\TL_09_파렛트,렉\불안전한 상태(UC)",
     r"D:\121.물류창고 내 작업 안전 데이터\01.데이터\1.Training\원천데이터\TS_09_파렛트,렉\불안전한 상태(UC)",
     "121_RK_UC_", mapping_121, "121", 10),
]

# 폴더 초기화
if os.path.exists(SAVE_DIR): shutil.rmtree(SAVE_DIR)
os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "labels"), exist_ok=True)

def imread_korean(path):
    try:
        with open(path, "rb") as f:
            return cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    except: return None

# === 2. 통합 전처리 루프 ===
total_saved = 0

for label_root, img_root, prefix, mapping, d_type, sampling in tasks:
    if not os.path.exists(label_root):
        print(f"❌ 경로 없음 스킵: {label_root}")
        continue

    # 하위 폴더의 모든 JSON 파일을 재귀적으로 탐색
    print(f"\n📂 {prefix} 데이터 스캔 중...")
    json_paths = glob.glob(os.path.join(label_root, "**", "*.json"), recursive=True)
    print(f"🔎 발견된 JSON: {len(json_paths)}개")

    # 이미지 파일 위치를 빠르게 찾기 위해 사전 인덱싱
    for i, j_path in enumerate(tqdm(json_paths, desc=f"Processing {prefix}")):
        if i % sampling != 0: continue
        
        try:
            with open(j_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            yolo_labels = []
            # [A] 데이터셋별 파싱
            if d_type == "107":
                meta = data.get('image') or {}
                w_orig, h_orig = meta.get('width', 1920), meta.get('height', 1080)
                img_name = meta.get('file_name', '')
                img_base = os.path.splitext(img_name)[0]
                annotations = data.get('annotations', [])
            else: # 121번
                w_orig, h_orig = data['Raw data Info.']['resolution']
                img_base = data['Source data Info.']['source_data_ID']
                img_name = img_base + ".jpg"
                annotations = data['Learning data info.']['annotation']

            # [B] 어노테이션 변환
            for ann in annotations:
                cat = str(ann.get('category_id') or ann.get('class_id') or "")
                if cat in mapping:
                    cid = mapping[cat]
                    coord = ann.get('bbox') or ann.get('coord')
                    if not coord: continue

                    # 좌표 형식 변환 (x_min, y_min, w, h로 통일)
                    if d_type == "107":
                        x_min, y_min, x_max, y_max = coord
                        w, h = x_max - x_min, y_max - y_min
                    else:
                        if isinstance(coord[0], list): # Polygon
                            pts = np.array(coord)
                            x_min, y_min = np.min(pts, axis=0)
                            x_max, y_max = np.max(pts, axis=0)
                            w, h = x_max - x_min, y_max - y_min
                        else: # Box
                            x_min, y_min, w, h = coord
                    
                    # YOLO 정규화
                    x_c = (x_min + w/2) / w_orig
                    y_c = (y_min + h/2) / h_orig
                    w_n = w / w_orig
                    h_n = h / h_orig

                    if 0 < x_c < 1 and 0 < y_c < 1 and w_n < 0.9:
                        yolo_labels.append(f"{cid} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

            # [C] 이미지 매칭 및 저장
            if yolo_labels:
                # 라벨 경로를 바탕으로 이미지 경로 추정 (AI-Hub 표준 구조 대응)
                relative_path = os.path.relpath(j_path, label_root)
                img_path = os.path.join(img_root, os.path.dirname(relative_path), img_name)
                
                # 위 경로에 없으면 파일명으로 다시 검색 (유연한 대응)
                if not os.path.exists(img_path):
                    # 파일명만 가지고 이미지 폴더 전체에서 찾기
                    search_pattern = os.path.join(img_root, "**", img_name)
                    found_imgs = glob.glob(search_pattern, recursive=True)
                    if found_imgs: img_path = found_imgs[0]
                    else: continue

                img = imread_korean(img_path)
                if img is not None:
                    save_id = f"{prefix}{img_base}"
                    cv2.imwrite(os.path.join(SAVE_DIR, "images", f"{save_id}.jpg"), 
                                cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
                    with open(os.path.join(SAVE_DIR, "labels", f"{save_id}.txt"), 'w') as f:
                        f.write("\n".join(yolo_labels))
                    total_saved += 1
        except:
            continue

print(f"\n✅ 전처리 완료! 총 {total_saved}세트 저장됨: {SAVE_DIR}")
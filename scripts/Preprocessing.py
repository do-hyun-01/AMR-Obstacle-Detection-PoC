import os
import json
import shutil
import random
from tqdm import tqdm

# 데이터셋 구축 및 분할 통합 스크립트
# 1. 경로 설정
IMG_ROOT = r"D:\121.로봇 관점 주행 영상 데이터\01.데이터\1.Training\1.원천데이터_230202_add\S(특수상황)\I(산업시설)\1.이미지"
LBL_ROOT = r"D:\121.로봇 관점 주행 영상 데이터\01.데이터\1.Training\2.라벨링데이터_230202_add\S(특수상황)\I(산업시설)"
DST_ROOT = r"D:\datasets\v5_original"

SPLIT_RATIO = 0.8
random.seed(42)

# S_I_1.json의 실제 'name'과 일치하는 매핑 (영문/한글 모두 대응)
category_map = {
    'moving object': 0, 'moving_object': 0, '카트': 0, '지게차': 0, '이동형': 0,
    'fixed object': 1, 'fixed_object': 1, '작업대': 1, '기둥': 1, '선반': 1,
    'person': 2, '사람': 2,
    'door': 3, '문': 3
}

def run_process():
    # 저장 폴더 생성
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DST_ROOT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DST_ROOT, split, "labels"), exist_ok=True)

    print(" 이미지 폴더 구조 스캔 중...")
    img_folder_map = {}
    for root, dirs, files in os.walk(IMG_ROOT):
        if os.path.basename(root) == "images":
            # images 폴더의 부모 폴더 이름 (예: S_I_001) 추출
            seq_name = os.path.basename(os.path.dirname(root))
            img_folder_map[seq_name] = root

    print(f" JSON 분석 및 매칭 중... (대상 시퀀스: {len(img_folder_map)}개)")
    all_data_tasks = []

    for root, dirs, files in os.walk(LBL_ROOT):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                # 파일명이 아닌 폴더 경로에서 S_I_xxx를 추출함
                path_parts = os.path.normpath(json_path).split(os.sep)
                # annotations 폴더의 부모 폴더가 S_I_xxx 임
                seq_name = next((p for p in reversed(path_parts) if p.startswith("S_I_") and not p.endswith(".json")), None)
                
                if not seq_name or seq_name not in img_folder_map:
                    continue

                img_dir = img_folder_map[seq_name]
                with open(json_path, 'r', encoding='utf-8') as f:
                    coco = json.load(f)

                # 카테고리 ID -> YOLO 인덱스 변환 맵 생성
                id_to_idx = {}
                for cat in coco.get('categories', []):
                    # 공백 및 특수문자 제거 후 매핑 대조
                    clean_name = cat['name'].lower().replace("_", " ")
                    idx = category_map.get(clean_name, -1)
                    # 만약 못 찾았다면 키워드 검색 시도
                    if idx == -1:
                        for k, v in category_map.items():
                            if k in clean_name:
                                idx = v; break
                    id_to_idx[cat['id']] = idx

                images_info = {img['id']: img for img in coco.get('images', [])}
                
                # 어노테이션 처리
                label_dict = {img_id: [] for img_id in images_info.keys()}
                for ann in coco.get('annotations', []):
                    yolo_idx = id_to_idx.get(ann['category_id'], -1)
                    if yolo_idx != -1 and ann['image_id'] in images_info:
                        img_meta = images_info[ann['image_id']]
                        # YOLO 좌표 정규화
                        dw, dh = 1.0 / img_meta['width'], 1.0 / img_meta['height']
                        x, y, w, h = ann['bbox'] # COCO: [x, y, w, h]
                        x_c = (x + w/2) * dw
                        y_c = (y + h/2) * dh
                        w_n = w * dw
                        h_n = h * dh
                        label_dict[ann['image_id']].append(f"{yolo_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                for img_id, labels in label_dict.items():
                    if labels:
                        img_file = images_info[img_id]['file_name']
                        src_path = os.path.join(img_dir, img_file)
                        if os.path.exists(src_path):
                            all_data_tasks.append({
                                'src': src_path,
                                'new_name': f"{seq_name}_{img_file}",
                                'labels': labels
                            })

    if not all_data_tasks:
        print(" 실패: 매칭된 데이터가 없습니다.")
        return

    print(f" 총 {len(all_data_tasks)}개 데이터 분할 및 저장 중...")
    random.shuffle(all_data_tasks)
    split_idx = int(len(all_data_tasks) * SPLIT_RATIO)

    for i, task in enumerate(tqdm(all_data_tasks, desc="데이터 복사")):
        split = 'train' if i < split_idx else 'val'
        # 이미지 복사
        shutil.copy2(task['src'], os.path.join(DST_ROOT, split, "images", task['new_name']))
        # 라벨 저장
        txt_name = os.path.splitext(task['new_name'])[0] + ".txt"
        with open(os.path.join(DST_ROOT, split, "labels", txt_name), 'w') as f:
            f.write("\n".join(task['labels']))

    print(f" 완료! [Train: {split_idx}, Val: {len(all_data_tasks)-split_idx}]")

if __name__ == "__main__":
    run_process()
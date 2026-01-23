import cv2
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

SRC_ROOT = r"D:\datasets\v5_original"
DST_ROOT = r"D:\datasets\v5_final"
TARGET_SIZE = 640

def process_file(file_info):
    src_path, dst_path, is_image = file_info
    
    # 목적지 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    if is_image:
        try:
            img = cv2.imread(src_path)
            if img is None: return
            
            # 비율 유지 리사이징 로직
            h, w = img.shape[:2]
            if h > w:
                new_h, new_w = TARGET_SIZE, int(TARGET_SIZE * w / h)
            else:
                new_h, new_w = int(TARGET_SIZE * h / w), TARGET_SIZE
                
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
    else:
        # 라벨 파일(.txt) 등은 그대로 복사
        shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    tasks = []
    print(" 파일 목록 스캔 중...")
    for root, dirs, files in os.walk(SRC_ROOT):
        for f in files:
            src_p = os.path.join(root, f)
            # 원본 구조를 유지하며 목적지 경로 생성
            rel_p = os.path.relpath(src_p, SRC_ROOT)
            dst_p = os.path.join(DST_ROOT, rel_p)
            
            is_img = f.lower().endswith(('.jpg', '.jpeg', '.png'))
            tasks.append((src_p, dst_p, is_img))

    print(f" 리사이징 시작 (대상: {len(tasks)}개)")
    # 서버의 16개 코어 전체 가용하여 병렬 처리
    with ProcessPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(process_file, tasks), total=len(tasks)))

    print(f"\n 완료! 새 데이터셋 경로: {DST_ROOT}")
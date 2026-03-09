import os
import shutil

def collect_cvat_data(src_path, base_dest_path):
    # 1. 대상 폴더 경로 설정 및 생성
    dest_img_dir = os.path.join(base_dest_path, 'images_cvat')
    dest_lbl_dir = os.path.join(base_dest_path, 'labels_cvat')

    # 폴더가 없으면 생성
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)

    # 2. 소스 폴더의 모든 파일 목록 가져오기
    all_files = os.listdir(src_path)
    count = 0

    print(f" 데이터 분류 시작: {src_path}")

    for f in all_files:
        # 라벨 파일(.txt)을 기준으로 검사
        if f.endswith('.txt'):
            lbl_path = os.path.join(src_path, f)
            
            # 파일 크기가 0보다 큰 경우(라벨링된 경우)만 진행
            if os.path.getsize(lbl_path) > 0:
                name = os.path.splitext(f)[0]
                
                # 이미지 파일 찾기 (jpg, jpeg, png 대응)
                img_exts = ['.jpg', '.jpeg', '.png']
                img_found = False
                
                for ext in img_exts:
                    img_name = name + ext
                    src_img_path = os.path.join(src_path, img_name)
                    
                    if os.path.exists(src_img_path):
                        # 이미지와 라벨을 새 폴더로 복사
                        shutil.copy(src_img_path, os.path.join(dest_img_dir, img_name))
                        shutil.copy(lbl_path, os.path.join(dest_lbl_dir, f))
                        img_found = True
                        count += 1
                        break
                
                if not img_found:
                    print(f" {f}에 해당하는 이미지 파일을 찾을 수 없습니다.")

    print(f" 분류 완료! 총 {count}세트의 데이터 추가 가능.")
    print(f" 이미지: {dest_img_dir}")
    print(f" 라벨: {dest_lbl_dir}")

# 실행
source = r'D:\datasets\dataset_video\obj_train_data'
destination_base = r'D:\datasets\v5_final\train'

collect_cvat_data(source, destination_base)


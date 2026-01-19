import os
import random
import shutil
from tqdm import tqdm

# === 설정 ===
base_path = r"C:\Users\Administrator\Desktop\AMR_Project\AMR_Dataset_Final"
train_ratio = 0.8

# 폴더 구조 생성
for split in ['train', 'val']:
    for sub in ['images', 'labels']:
        os.makedirs(os.path.join(base_path, split, sub), exist_ok=True)

<<<<<<< HEAD
# === 수정된 파일 목록 확보 로직 ===
img_dir = os.path.join(base_path, "images")
=======
# 파일 목록 확보
image_files = [f for f in os.listdir(os.path.join(base_path, "images")) if f.endswith('.JPG')]
random.shuffle(image_files)
>>>>>>> 1f286aee6c0092424a395f8c40f9a9cccc591c56

# 확장자 대소문자 구분 없이 .jpg, .jpeg, .png 모두 검색
image_files = [f for f in os.listdir(img_dir) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

<<<<<<< HEAD
if len(image_files) == 0:
    print(f"❌ 에러: {img_dir} 폴더에서 이미지 파일을 찾을 수 없습니다.")
    print(f"현재 폴더 내 파일 샘플: {os.listdir(img_dir)[:5]}") # 디버깅용 출력
else:
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
=======
def move_files(files, split_name):
    print(f"{split_name} 데이터 이동 중...")
    for f in tqdm(files):
        # 이미지 이동
        shutil.move(os.path.join(base_path, "images", f), 
                    os.path.join(base_path, split_name, "images", f))
        # 라벨 이동
        label_f = f.replace('.JPG', '.txt')
        label_src = os.path.join(base_path, "labels", label_f)
        if os.path.exists(label_src):
            shutil.move(label_src, os.path.join(base_path, split_name, "labels", label_f))
>>>>>>> 1f286aee6c0092424a395f8c40f9a9cccc591c56

    def move_files(files, split_name):
        print(f"\n📂 {split_name} 데이터 이동 중 ({len(files)}개)...")
        for f in tqdm(files):
            # 이미지 이동
            shutil.move(os.path.join(base_path, "images", f), 
                        os.path.join(base_path, split_name, "images", f))
            # 라벨 이동 (확장자만 .txt로 변경)
            label_f = os.path.splitext(f)[0] + ".txt"
            label_src = os.path.join(base_path, "labels", label_f)
            if os.path.exists(label_src):
                shutil.move(label_src, os.path.join(base_path, split_name, "labels", label_f))

    move_files(train_files, 'train')
    move_files(val_files, 'val')

    print(f"\n✅ 분할 완료! Train: {len(train_files)}개 / Val: {len(val_files)}개")
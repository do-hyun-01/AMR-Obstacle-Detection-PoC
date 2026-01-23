from ultralytics import YOLO

def train_model():
    # 1. 모델 로드
    model = YOLO(r'D:\datasets\v5_final\AMR_Project\yolo26_resized\weights\best.pt')

    # 2. 학습 실행
    model.train(
        data=r'D:\datasets\v5_final\data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.0005,
        optimizer='MuSGD',
        device=0,
        project='AMR_Project',
        name='yolo26_final_field_adapted',
        workers=4
    )

if __name__ == '__main__':
    train_model()
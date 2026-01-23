import os
import glob

# 클래스 ID 정의
MOVING_ID = 0
FIXED_ID = 1

def yolo_to_xyxy(box):
    """중심 좌표 형식을 [x1, y1, x2, y2] 형식으로 변환"""
    cls, x, y, w, h = box
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    return [cls, x1, y1, x2, y2]

def xyxy_to_yolo(box):
    """[x1, y1, x2, y2] 형식을 다시 YOLO 형식으로 변환"""
    cls, x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x, y = x1 + w/2, y1 + h/2
    return [cls, x, y, w, h]

def is_overlapping(box_a, box_b):
    """두 박스가 겹치는지 확인 (xyxy 형식)"""
    _, a_x1, a_y1, a_x2, a_y2 = box_a
    _, b_x1, b_y1, b_x2, b_y2 = box_b
    return not (a_x2 < b_x1 or a_x1 > b_x2 or a_y2 < b_y1 or a_y1 > b_y2)

def merge_labels(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = glob.glob(os.path.join(input_path, "*.txt"))
    
    for file in files:
        with open(file, 'r') as f:
            lines = [list(map(float, line.split())) for line in f.readlines()]

        if not lines: continue

        # xyxy 형식으로 변환
        boxes = [yolo_to_xyxy(line) for line in lines]
        moving_boxes = [b for b in boxes if b[0] == MOVING_ID]
        fixed_boxes = [b for b in boxes if b[0] == FIXED_ID]
        other_boxes = [b for b in boxes if b[0] not in [MOVING_ID, FIXED_ID]]

        final_moving = []
        used_fixed_indices = set()

        # Moving 객체 기준으로 겹치는 Fixed 객체 통합
        for m_box in moving_boxes:
            current_m = m_box[:]
            for i, f_box in enumerate(fixed_boxes):
                if is_overlapping(current_m, f_box):
                    # 좌표 확장: Min/Max 계산
                    current_m[1] = min(current_m[1], f_box[1]) # x1
                    current_m[2] = min(current_m[2], f_box[2]) # y1
                    current_m[3] = max(current_m[3], f_box[3]) # x2
                    current_m[4] = max(current_m[4], f_box[4]) # y2
                    used_fixed_indices.add(i)
            final_moving.append(current_m)

        # 통합되지 않은 나머지 고정 객체들
        remaining_fixed = [f for i, f in enumerate(fixed_boxes) if i not in used_fixed_indices]
        
        # 전체 합치기 및 다시 YOLO 형식으로 변환
        all_final = [xyxy_to_yolo(b) for b in final_moving + remaining_fixed + other_boxes]

        # 파일 저장
        new_file_path = os.path.join(output_path, os.path.basename(file))
        with open(new_file_path, 'w') as f:
            for b in all_final:
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    print(f" 병합 완료: {len(files)}개 파일 처리되었습니다.")

# 실행
input_labels = r"D:\datasets\v5_final\train\labels"
output_labels = r"D:\datasets\v5_final\train\labels_merged"
merge_labels(input_labels, output_labels)
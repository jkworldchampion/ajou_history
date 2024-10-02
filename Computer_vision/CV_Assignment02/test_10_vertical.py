import numpy as np
import cv2 as cv
import os
import evaluation as eval

def apply_vertical_filter(image):
    """
    흰-검-흰 패턴이 발견되면 검은색 픽셀을 흰색으로 바꾸는 필터
    """
    # 이미지의 높이와 너비를 가져옵니다.
    height, width = image.shape
    
    # 각 열마다 (x축) 흰-검-흰 패턴을 탐색
    for x in range(width):
        for y in range(1, height - 1):
            # 현재 픽셀 y와 위 아래 픽셀을 비교
            if image[y-1, x] == 255 and image[y, x] == 0 and image[y+1, x] == 255:
                # 흰-검-흰 패턴을 발견하면 가운데 검은 픽셀을 흰색으로 변경
                image[y, x] = 255

    return image

def main(default=19, corner=60):

    ##### Set threshold
    threshold_corner = corner  # 왼쪽 위 1/4 부분의 높은 threshold 설정
    threshold_default = default  # 나머지 부분의 기본 threshold 설정
    print('threshold_corner:', threshold_corner)
    print('threshold_default:', threshold_default)

    ##### Set path
    input_path = './input_image'    # input path
    gt_path = './groundtruth'       # groundtruth path
    result_path = './result'        # result path
    background_save_path = './background.png'  # 배경 이미지를 저장할 경로
    score_file_path = './score.txt'  # 점수를 저장할 파일 경로

    ##### load input
    input_img = [img for img in sorted(os.listdir(input_path)) if img.endswith(".jpg")]

    ##### first frame and initialize variables
    frame_current = cv.imread(os.path.join(input_path, input_img[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

    ##### Initialize background stack for the first 470 frames
    background_stack = []  # 각 프레임을 쌓아둘 리스트 초기화
    background_img = None  # 배경 이미지 초기화

    ##### background subtraction loop
    for image_idx in range(len(input_img)):

        ##### 배경을 만들기 위해 470개의 프레임을 쌓음
        if image_idx < 470:
            background_stack.append(frame_current_gray)  # 프레임을 리스트에 추가
            if image_idx == 469:  # 470번째 프레임에서 중위값 배경 생성
                background_stack_np = np.stack(background_stack, axis=2)  # (H, W, 470)로 쌓음
                background_img = np.median(background_stack_np, axis=2)  # 각 픽셀에 대한 중위값 계산

                ##### 중위값으로 만들어진 background_img를 저장
                cv.imwrite(background_save_path, background_img.astype(np.uint8))

        ##### If background_img is initialized, calculate the difference
        if background_img is not None:
            ##### calculate foreground region based on background
            diff = frame_current_gray - background_img
            diff_abs = np.abs(diff).astype(np.float64)

            ##### 이미지의 크기를 얻음
            h, w = frame_current_gray.shape

            ##### 왼쪽 위 1/4 부분에 대해 낮은 threshold 적용
            corner_diff_abs = diff_abs[:h // 2, :w // 2]  # 왼쪽 위 1/4 영역
            frame_diff_corner = np.where(corner_diff_abs > threshold_corner, 1.0, 0.0)

            ##### 나머지 부분에 대해 기본 threshold 적용
            rest_diff_abs_top = diff_abs[:h // 2, w // 2:]  # 위쪽 오른쪽 절반
            rest_diff_abs_bottom_left = diff_abs[h // 2:, :w // 2]  # 아래쪽 왼쪽 절반
            rest_diff_abs_bottom_right = diff_abs[h // 2:, w // 2:]  # 아래쪽 오른쪽 절반

            frame_diff_rest_top = np.where(rest_diff_abs_top > threshold_default, 1.0, 0.0)
            frame_diff_rest_bottom_left = np.where(rest_diff_abs_bottom_left > threshold_default, 1.0, 0.0)
            frame_diff_rest_bottom_right = np.where(rest_diff_abs_bottom_right > threshold_default, 1.0, 0.0)

            ##### 다시 하나의 이미지로 결합
            top_combined = np.hstack((frame_diff_corner, frame_diff_rest_top))
            bottom_combined = np.hstack((frame_diff_rest_bottom_left, frame_diff_rest_bottom_right))
            frame_diff = np.vstack((top_combined, bottom_combined))

            ##### 왼쪽 상단 거꾸로 삼각형 영역을 masking
            triangle_mask = np.ones((h, w), dtype=np.float64)  # 기본적으로 1로 설정된 마스크
            for y in range(h // 2, 0, -1):  # 위쪽 1/2 높이만큼 적용
                for x in range(w // 2, 0, -1):  # 왼쪽 1/2 너비만큼 적용
                    if x <= (w // 2) * ((h // 2 - y) / (h // 2)):  # 거꾸로 삼각형 내부
                        triangle_mask[y, x] = 0  # 삼각형 영역을 0으로 설정하여 검출되지 않도록 함

            ##### 마스크를 적용하여 해당 삼각형 영역을 검출하지 않도록 설정
            frame_diff_masked = np.multiply(frame_diff, triangle_mask)

            ##### apply mask to current frame
            current_gray_masked = np.multiply(frame_current_gray, frame_diff_masked)
            current_gray_masked_mk2 = np.where(current_gray_masked > 0, 255.0, 0.0)

            ##### Apply median filter to remove salt and pepper noise
            result = cv.medianBlur(current_gray_masked_mk2.astype(np.uint8), 3)  # 3x3 커널 사용

            ##### Apply the vertical filter
            result = apply_vertical_filter(result)

            ##### save result file
            cv.imwrite(os.path.join(result_path, 'result%06d.png' % (image_idx + 1)), result)

        ##### read next frame
        if image_idx < len(input_img) - 1:
            frame_current = cv.imread(os.path.join(input_path, input_img[image_idx + 1]))
            frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

    ##### evaluation result
    recall, precision, f1_score = eval.cal_result(gt_path, result_path)

    ##### Score 저장 및 비교

    # 현재 계산된 F1 Score를 가져옵니다.
    current_f1_score = f1_score

    # score.txt 파일이 존재하는지 확인합니다.
    if os.path.exists(score_file_path):
        # 파일이 존재하면 내용을 읽습니다.
        with open(score_file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                # 기존에 저장된 F1 Score를 가져옵니다.
                existing_f1_score = float(lines[2].split(':')[1].strip())
            else:
                existing_f1_score = -1  # 파일이 비어 있으면 -1로 설정
    else:
        existing_f1_score = -1  # 파일이 없으면 -1로 설정

    # 현재 F1 Score가 기존보다 높으면 파일을 업데이트합니다.
    if current_f1_score > existing_f1_score:
        with open(score_file_path, 'w') as f:
            f.write(f'threshold_corner: {threshold_corner}\n')
            f.write(f'threshold_default: {threshold_default}\n')
            f.write(f'F1 Score: {f1_score}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'Precision: {precision}\n')
        print('New best F1 Score found and saved to score.txt')
    else:
        print('Existing F1 Score is higher. No update to score.txt')

if __name__ == '__main__':
    #### main(default, corner)
    main(35, 84)
    main(20, 60)

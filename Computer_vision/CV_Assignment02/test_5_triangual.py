import numpy as np
import cv2 as cv
import os
import evaluation as eval

###############################################################
##### This code has been tested in Python 3.6 environment #####
###############################################################

def main():

    ##### Set threshold
    threshold_triangle = 50  # 삼각형 부분의 낮은 threshold 설정
    threshold_default = 30  # 나머지 부분의 기본 threshold 설정
    print('threshold_triangle:', threshold_triangle)
    print('threshold_default:', threshold_default)

    ##### Set path
    input_path = './input_image'    # input path
    gt_path = './groundtruth'       # groundtruth path
    result_path = './result'        # result path

    ##### load input
    input_img = [img for img in sorted(os.listdir(input_path)) if img.endswith(".jpg")]

    ##### first frame and initialize variables
    frame_current = cv.imread(os.path.join(input_path, input_img[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

    ##### Initialize background sum for the first 470 frames
    background_sum_img = np.zeros(frame_current_gray.shape, dtype=np.float64)
    background_img = None  # 배경 이미지 초기화
    
    ##### background subtraction loop
    for image_idx in range(len(input_img)):

        ##### print current image index
        if image_idx % 170 == 0:
            print('Processing %d%%' % (image_idx // 17))

        ##### 470개의 이미지에 대해서 단순한 중첩을 통해 배경을 만들어내는 방법
        if image_idx < 470:
            background_sum_img += frame_current_gray
            if image_idx == 469:  # 470번째 프레임에서 배경 모델을 만듦
                background_img = background_sum_img / 470

        ##### If background_img is initialized, calculate the difference
        if background_img is not None:
            ##### calculate foreground region based on background
            diff = frame_current_gray - background_img
            diff_abs = np.abs(diff).astype(np.float64)

            ##### 이미지의 크기를 얻음
            h, w = frame_current_gray.shape

            ##### 삼각형 마스크 생성 (왼쪽 위 1/2와 위쪽 1/2 지점을 잇는 삼각형)
            triangle_mask = np.zeros((h, w), dtype=np.float64)
            for y in range(h // 2, 0, -1):  # 위쪽 1/2 지점까지만 적용
                for x in range(w // 2, 0, -1):  # 왼쪽 1/2 지점까지만 적용
                    if x <= (w // 2) * (y / (h // 2)):  # 삼각형 내부 좌표 계산
                        triangle_mask[y, x] = 1.0

            ##### 삼각형 부분에 대해서는 낮은 threshold 적용
            diff_triangle = np.multiply(diff_abs, triangle_mask)
            frame_diff_triangle = np.where(diff_triangle > threshold_triangle, 1.0, 0.0)

            ##### 나머지 부분에 대해서는 기본 threshold 적용
            diff_rest = np.multiply(diff_abs, 1.0 - triangle_mask)
            frame_diff_rest = np.where(diff_rest > threshold_default, 1.0, 0.0)

            ##### 삼각형 부분과 나머지 부분 결합
            frame_diff = frame_diff_triangle + frame_diff_rest

            ##### apply mask to current frame
            current_gray_masked = np.multiply(frame_current_gray, frame_diff)
            current_gray_masked_mk2 = np.where(current_gray_masked > 0, 255.0, 0.0)

            ##### final result
            result = current_gray_masked_mk2.astype(np.uint8)

            ##### save result file
            cv.imwrite(os.path.join(result_path, 'result%06d.png' % (image_idx + 1)), result)

        ##### read next frame
        if image_idx < len(input_img) - 1:
            frame_current = cv.imread(os.path.join(input_path, input_img[image_idx + 1]))
            frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        ##### If you want to stop, press ESC key
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    ##### evaluation result
    eval.cal_result(gt_path, result_path)

if __name__ == '__main__':
    main()

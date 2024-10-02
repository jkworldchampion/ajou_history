import numpy as np
import cv2 as cv
import os
import evaluation as eval
from scipy.ndimage import binary_erosion, binary_dilation

def main():

    ##### Set threshold
    threshold_corner = 30  # 왼쪽 위 1/4 부분의 높은 threshold 설정
    threshold_default = 20  # 나머지 부분의 기본 threshold 설정
    print('threshold_corner:', threshold_corner)
    print('threshold_default:', threshold_default)

    ##### Set path
    input_path = './input_image'    # input path
    gt_path = './groundtruth'       # groundtruth path
    result_path = './result'        # result path
    background_save_path = './background.png'  # 배경 이미지를 저장할 경로

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

        ##### print current image index
        if image_idx % 170 == 0:
            print('Processing %d%%' % (image_idx // 17))

        ##### 배경을 만들기 위해 470개의 프레임을 쌓음
        if image_idx < 470:
            background_stack.append(frame_current_gray)  # 프레임을 리스트에 추가
            if image_idx == 469:  # 470번째 프레임에서 중위값 배경 생성
                background_stack_np = np.stack(background_stack, axis=2)  # (H, W, 470)로 쌓음
                background_img = np.median(background_stack_np, axis=2)  # 각 픽셀에 대한 중위값 계산

                ##### 중위값으로 만들어진 background_img를 저장
                cv.imwrite(background_save_path, background_img.astype(np.uint8))
                print(f'Background image saved at {background_save_path}')

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

            ##### 마스킹 단계 제거 또는 조정 (여기서는 제거)
            frame_diff_masked = frame_diff

            ##### 이진 이미지로 변환
            frame_diff_masked_binary = (frame_diff_masked * 255).astype(np.uint8)

            ##### 형태학적 연산을 직접 구현

            # 5x5 타원형 구조 요소 생성
            def create_elliptical_kernel(size):
                y, x = np.ogrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
                mask = x**2 + y**2 <= (size//2)**2
                kernel = mask.astype(np.uint8)
                return kernel

            kernel_size = 5
            kernel = create_elliptical_kernel(kernel_size)

            # 패딩 크기 계산
            pad_size = kernel_size // 2

            # 이진 이미지를 0과 1로 변환
            binary_image = frame_diff_masked_binary // 255

            # 침식 연산 구현
            def erosion(image, kernel):
                padded_image = np.pad(image, pad_size, mode='constant', constant_values=1)
                eroded_image = np.zeros_like(image)
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        region = padded_image[i:i+kernel_size, j:j+kernel_size]
                        eroded_image[i, j] = np.min(region[kernel == 1])
                return eroded_image

            # 팽창 연산 구현
            def dilation(image, kernel):
                padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
                dilated_image = np.zeros_like(image)
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        region = padded_image[i:i+kernel_size, j:j+kernel_size]
                        dilated_image[i, j] = np.max(region[kernel == 1])
                return dilated_image

            # 열림 연산 (침식 후 팽창)
            eroded = erosion(binary_image, kernel)
            opened = dilation(eroded, kernel)

            # 닫힘 연산 (팽창 후 침식)
            dilated = dilation(opened, kernel)
            closed = erosion(dilated, kernel)

            # 결과 이미지를 0과 255로 변환
            result = (closed * 255).astype(np.uint8)

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

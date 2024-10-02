import numpy as np
import cv2 as cv
import os
import evaluation as eval

###############################################################
##### This code has been tested in Python 3.6 environment #####
###############################################################

def main():

    ##### Set threshold
    threshold = 30  # 이 값을 변하는 값으로 하여 성능을 올릴 수 있음

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
        if image_idx%170 == 0:
            print('Processing %d%%' % (image_idx//17))

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

            ##### make mask by applying threshold
            frame_diff = np.where(diff_abs > threshold, 1.0, 0.0)

            ##### apply mask to current frame
            current_gray_masked = np.multiply(frame_current_gray, frame_diff)
            current_gray_masked_mk2 = np.where(current_gray_masked > 0, 255.0, 0.0)

            ##### final result
            result = current_gray_masked_mk2.astype(np.uint8)
            #cv.imshow('result', result) # colab does not support cv.imshow

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

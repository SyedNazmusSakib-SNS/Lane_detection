import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# selected threshold to highlight yellow lines
yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])

def thresh_frame_in_HSV(frame, min_values, max_values, verbose=False):
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_th_ok = np.all(HSV > min_values, axis=2)
    max_th_ok = np.all(HSV < max_values, axis=2)
    out = np.logical_and(min_th_ok, max_th_ok)
    if verbose:
        plt.imshow(out, cmap='gray')
        plt.show()
    return out

def thresh_frame_sobel(frame, kernel_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)
    return sobel_mag.astype(bool)

def get_binary_from_equalized_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq_global = cv2.equalizeHist(gray)
    _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
    return th

def binarize(img, verbose=False):
    h, w = img.shape[:2]
    binary = np.zeros(shape=(h, w), dtype=np.uint8)
    HSV_yellow_mask = thresh_frame_in_HSV(img, yellow_HSV_th_min, yellow_HSV_th_max, verbose=False)
    binary = np.logical_or(binary, HSV_yellow_mask)
    eq_white_mask = get_binary_from_equalized_grayscale(img)
    binary = np.logical_or(binary, eq_white_mask)
    sobel_mask = thresh_frame_sobel(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if verbose:
        f, ax = plt.subplots(2, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('Input Frame')
        ax[0, 0].set_axis_off()
        ax[0, 1].imshow(eq_white_mask, cmap='gray')
        ax[0, 1].set_title('White Mask')
        ax[0, 1].set_axis_off()
        ax[0, 2].imshow(HSV_yellow_mask, cmap='gray')
        ax[0, 2].set_title('Yellow Mask')
        ax[0, 2].set_axis_off()
        ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('Sobel Mask')
        ax[1, 0].set_axis_off()
        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('Before Closure')
        ax[1, 1].set_axis_off()
        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('After Closure')
        ax[1, 2].set_axis_off()
        plt.show()
    return closing

if __name__ == '__main__':
    test_images = glob.glob('test_images/*.jpg')
    for test_image in test_images:
        img = cv2.imread(test_image)
        binarize(img=img, verbose=True)

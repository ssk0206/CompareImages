from skimage.measure import compare_ssim, compare_psnr
import cv2
import time
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

NUM_OF_FILES = 25
# METHOD = 'psnr'
METHOD = 'ssim'

def evaluate(img1, img2, method):
    val = 0
    if method == 'ssim':
        val = compare_ssim(img1, img2, multichannel=True)
    elif method == 'psnr':
        val = compare_psnr(img1, img2)
    return val

def show_graph(ssim_aug, ssim_pix):
    left = np.arange(len(ssim_aug))  # numpyで横軸を設定
    labels = []
    file_list = get_file_name_list("./data/pix/*.png")
    for i, p in enumerate(file_list):
        if i%2 == 0:
            labels.append(p[11:-9])

    width = 0.3

    plt.xlabel('Images')
    plt.ylabel('Score')
    plt.title(METHOD)

    plt.bar(left, ssim_pix, color='b', width=width, align='center')
    plt.bar(left+width, ssim_aug, color='r', width=width, align='center')

    plt.xticks(left + width/2, labels, fontsize=6)
    plt.show()

def get_file_name_list(file_path):
    file_list = glob.glob(file_path)
    return sorted(file_list)


def main():
    file_list_augmented = get_file_name_list("./data/aug/*.png")
    file_list_pix2pix = get_file_name_list("./data/pix/*.png")

    aug_list = []
    pix_list = []

    for i in range(NUM_OF_FILES):
        augmented_fake_img = cv2.imread(file_list_augmented[2*i])
        augmented_real_img = cv2.imread(file_list_augmented[2*i+1])
        pix2pix_fake_img = cv2.imread(file_list_pix2pix[2*i])
        pix2pix_real_img = cv2.imread(file_list_augmented[2*i+1]) # ground_truthはaugの方の画像を使用する

        aug_list.append(evaluate(augmented_real_img, augmented_fake_img, METHOD))
        pix_list.append(evaluate(pix2pix_real_img, pix2pix_fake_img, METHOD))

    print('aug: ', sum(aug_list)/NUM_OF_FILES)
    print('pix: ', sum(pix_list)/NUM_OF_FILES)

    show_graph(aug_list, pix_list)


if __name__ == '__main__':
    main()
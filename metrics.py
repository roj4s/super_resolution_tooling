from skimage import metrics as skm
from skimage.io import imread
import argparse
import os

def prepare_images_pair(img1_addr, img2_addr):
    img1 = imread(img1_addr)
    img2 = imread(img2_addr)
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_height,:min_width, :]
    img2 = img2[:min_height,:min_width, :]
    return img1, img2

def psnr_from_files(img1_addr, img2_addr):
    img1, img2 = prepare_images_pair(img1_addr, img2_addr)
    return metrics.peak_signal_noise_ratio(img1, img2)

def metrics(folder_1, folder_2):
    '''
        Given two folders (e.g generated images vs original images),
        this script computes Averaged Peak Signal Noise Ratio (psnr) and
        Structural Similarity (ssim) for each pair
        of images (same name in both folders).
    '''
    img_names = os.listdir(folder_1)
    psnr = 0
    ssim = 0
    for img_name in img_names:
        img1_addr = os.path.join(folder_1, img_name)
        img2_addr = os.path.join(folder_2, img_name)
        img1, img2 = prepare_images_pair(img1_addr, img2_addr)
        psnr += skm.peak_signal_noise_ratio(img1, img2)
        ssim += skm.structural_similarity(img1, img2, multichannel=True)

    return (psnr / len(img_names), ssim / len(img_names))

if __name__ == "__main__":
    import sys
    psnr, ssim = metrics(sys.argv[1], sys.argv[2])
    print(f"SSIM: {ssim}")
    print(f"PSNR: {psnr}")

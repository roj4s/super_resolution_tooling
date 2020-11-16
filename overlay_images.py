from bicubic_interpolation import upsampling_lr_to_hr
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import os
import argparse


def overlay(fg_img_addr, bg_img_addr, output_folder=None):
    bg_img = cv2.imread(bg_img_addr)
    fg_img = cv2.imread(fg_img_addr)

    new_img = cv2.addWeighted(bg_img, 0.7, fg_img, 0.3, 0)

    if output_folder is not None:
        output_addr = os.path.join(output_folder, f"overlay_{bg_img}")
        cv2.imwrite(output_addr, new_img)

    return new_img

def overlay_folder(aligned_folder, ref_folder, output_folder=None):
    img_names = os.listdir(ref_folder)
    for imgn in tqdm(img_names):
        alig_img_addr = os.path.join(aligned_folder, imgn)
        if not os.path.exists(alig_img_addr):
            continue
        ref_img_addr = os.path.join(ref_folder, imgn)
        si_image = overlay(alig_img_addr, ref_img_addr)

        if output_folder is not None:
            output_addr = os.path.join(output_folder, f"overlay_{imgn}")
            cv2.imwrite(output_addr, si_image)

        yield si_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay images for alignment' \
                                     'visual check')
    parser.add_argument('foreground', type=str, help='Foreground image')
    parser.add_argument('background', type=str, help='Background image')
    parser.add_argument('--isdir', help='Foreground and background'\
                        ' are directories', action='store_true', default=False)
    parser.add_argument('--output', help='Optional directory to output results', type=str,
                        default=None)
    parser.add_argument('--show', help='Show overlay output', action='store_true', default=False)

    args = parser.parse_args()
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    if args.output is None:
        args.show = True

    imgs = []
    if args.isdir:
        imgs = [img for img in overlay_folder(args.foreground, args.background,
                                  args.output)]
    else:
        imgs.append(overlay(args.foreground, args.background, args.output))

    if args.show:
        for img in imgs:
            cv2.imshow('overlay', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

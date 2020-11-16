from bicubic_interpolation import upsampling_lr_to_hr
from matplotlib import pyplot as plt
import cv2


def overlay(fg_img_addr, bg_img_addr):
    bg_img = cv2.imread(bg_img_addr)
    fg_img = cv2.imread(fg_img_addr)

    new_img = cv2.addWeighted(bg_img, 0.7, fg_img, 0.3, 0)

    return new_img


if __name__ == "__main__":
    import sys
    lr_addr = sys.argv[1]
    hr_addr = sys.argv[2]
    output = sys.argv[3]
    lr_img = upsampling_lr_to_hr(lr_addr, hr_addr)
    lr_up_addr = '/tmp/upsampled_lr.png'
    lr_img.save(lr_up_addr)
    si_image = overlay(lr_up_addr, hr_addr)
    cv2.imwrite(output, si_image)
    cv2.imshow('overlay', si_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

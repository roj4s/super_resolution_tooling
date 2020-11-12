from matplotlib import pyplot as plt
import numpy as np
import time
from bicubic_interpolation import bicubic_interp
from sklearn.metrics import mean_squared_error
import silx
from silx.image import sift
from silx.test.utils import utilstest
from PIL import Image
import os
import logging as log
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"



def kp_rmse(img_path_lr, img_path_hr, device='CPU', plot=False):
    #path_lr = utilstest.getfile(img_path_lr)
    #path_hr = utilstest.getfile(img_path_hr)

    lr = np.asarray(Image.open(img_path_lr))
    hr = np.asarray(Image.open(img_path_hr))

    log.warning(f"LR shape: {np.shape(lr)} ")
    log.warning(f"HR shape: {np.shape(hr)} ")

    sift_ocl_lr = sift.SiftPlan(template=lr, devicetype=device)
    sift_ocl_hr = sift.SiftPlan(template=hr, devicetype=device)

    t = time.time()
    kps_lr = sift_ocl_lr(lr)
    t = time.time() - t
    log.warning(f"Got {len(kps_lr)} keypoints for lr in {t:.2f} secs")

    t = time.time()
    kps_hr = sift_ocl_hr(hr)
    t = time.time() - t
    log.warning(f"Got {len(kps_hr)} keypoints for hr in {t:.2f} secs")

    mp = sift.MatchPlan()
    match = mp(kps_hr, kps_lr)
    log.warning(f"Got {match.shape[0]} matches")

    if match.shape[0] == 0:
        return (-1, 0)

    lr_shape = np.shape(lr)
    hr_shape = np.shape(hr)

    mp_hr = np.column_stack(([match[:,0].x, match[:,0].y]))
    mp_lr = np.column_stack(([match[:,1].x, match[:,1].y]))

    mp_hr_norm = np.column_stack(([mp_hr[:,0] / hr_shape[1], mp_hr[:,1]
                                   / hr_shape[0]]))

    mp_lr_norm = np.column_stack(([mp_lr[:,0] / lr_shape[1], mp_lr[:,1]
                                   / lr_shape[0]]))

    rmse = mean_squared_error(mp_hr_norm, mp_lr_norm, squared=False)
    log.warning(f"RMSE: {rmse}")

    if plot:
        fig, (ax_lr, ax_hr) = plt.subplots(1, 2)
        ax_hr.imshow(hr)
        ax_hr.plot(match[:,0].x, match[:,0].y, ".g", markersize=2)

        ax_lr.imshow(lr)
        ax_lr.plot(match[:,1].x, match[:,1].y, ".g", markersize=2)
        plt.show()

    return (rmse, match.shape[0])

if __name__ == "__main__":
    loglevel = os.environ.get("LOGLEVEL", "WARNING")
    log.basicConfig(level=loglevel)
    root_hr = "/home/rojas/datasets/real-world-super-resolution/Test_x2/test_HR"
    root_lr = "/home/rojas/datasets/real-world-super-resolution/Test_x2/test_LR"
    for img_name in os.listdir(root_lr):
        img_name = img_name.split("_x1")[0]
        img_path_lr = os.path.join(root_lr, f"{img_name}_x1.png")
        img_path_hr = os.path.join(root_hr, f"{img_name}_x2.png")
        kp_rmse(img_path_lr, img_path_hr, plot=True)

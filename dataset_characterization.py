import pandas as pd
from keypoints_rmse import kp_rmse
from metrics import psnr_ssim_upsample
from PIL import Image
from dataset_iterator import SRDatasetIterator
import os


def metrics_from_dataset(dataset_iterator, output_csv_addr):

    columns = ('image_name', 'lr_path', 'hr_path', 'scale', 'rmse', 'psnr',
        'ssim',
        'width_lr',
        'height_lr',
        'width_hr',
        'height_hr',
        'match_keypoints')
    columns_str = ",".join(columns)

    if not os.path.exists(output_csv_addr):
        with open(output_csv_addr, 'wt') as f:
            f.write(f"{columns_str}\n")

    od = None
    if os.path.exists(output_csv_addr):
        od = pd.read_csv(output_csv_addr)

    for pdd in dataset_iterator.get_pair():

        try:
            if od is not None:
                if od[od['lr_path'] == pdd['lr']].shape[0]:
                    print(f"Skipping {pdd['image_name']}_{pdd['scale']}")
                    continue

            (rmse, matches) = kp_rmse(pdd['lr'], pdd['hr'])
            (psnr, ssim) = psnr_ssim_upsample(pdd['lr'], pdd['hr'])
            (lr_img_width, lr_img_height) = Image.open(pdd['lr']).size
            (hr_img_width, hr_img_height) = Image.open(pdd['hr']).size

            data = {'image_name': pdd['image_name'],
                    'lr_path': pdd['lr'],
                    'hr_path': pdd['hr'],
                    'scale': pdd['scale'],
                    'rmse': rmse,
                    'psnr': psnr,
                    'ssim': ssim,
                    'width_lr': lr_img_width,
                    'height_lr': lr_img_height,
                    'width_hr': hr_img_width,
                    'height_hr': hr_img_height,
                    'match_keypoints': matches,
                    'is_training': pdd['is_train'],
                    'is_testing': pdd['is_test'],
                }

            data_str = ",".join([str(data[c]) for c in columns])
            with open(output_csv_addr, 'at') as f:
                f.write(f"{data_str}\n")
        except Exception as e:
            print("ERROR")
            print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get dataset metrics for'\
                                     'characterization')
    parser.add_argument('dataset_addr', type=str, help='Dataset address')
    parser.add_argument('output', type=str, help='Output address for csv'\
                        'with metrics results')
    parser.add_argument('--hr-subfolder', type=str, help='Use images in this subfolder as'\
                        'HR')
    parser.add_argument('--lr-subfolder', type=str, help='Use images in this subfolder as'\
                        'LR')


    args = parser.parse_args()
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    it = SRDatasetIterator(args.data_addr, HR_subfolder=args.hr_subfolder,
                           LR_subfolder=args.lr_subfolder)
    metrics_from_dataset(it, args.output)


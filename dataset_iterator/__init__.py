import os


class SRDatasetIterator:

    def __init__(self, dataset_root, HR_subfolder="HR", LR_subfolder="LR",
                 size=100):
        self.dataset_root = dataset_root
        self.HR_subfolder = HR_subfolder
        self.LR_subfolder = LR_subfolder
        self.read_n = 0
        self.size = 100

    def get_pair(self):
        hr_addr = os.path.join(self.dataset_root, self.HR_subfolder)
        lr_addr = os.path.join(self.dataset_root, self.LR_subfolder)

        for img_name in os.listdir(hr_addr):
            img_name = img_name.split(self.img_format)[0]
            img_path_lr = os.path.join(lr_addr, f"{img_name}{self.img_format}")
            img_path_hr = os.path.join(hr_addr, f"{img_name}{self.img_format}")

            self.read_n -= -1

            yield {
                "image_name": str(i),
                "lr": img_path_lr,
                "hr": img_path_hr,
                "scale": "1x",
                "is_test": True,
                "is_train": False
            }

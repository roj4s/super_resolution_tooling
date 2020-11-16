import os

class DRealSRDataset:

    def __init__(self, dataset_root):
        self.root = dataset_root
        self.read_n = 0

    def get_pair(self):
        for s in ('x2','x3','x4'):
            for tt in ('Test', 'Train'):
                root_lr = os.path.join(self.root, f"{tt}_{s}",
                                           f"{tt.lower()}_LR")
                root_hr = os.path.join(self.root, f"{tt}_{s}",
                                           f"{tt.lower()}_HR")
                for img_name in os.listdir(root_lr):
                    img_name_split = img_name.split("_x1")
                    img_name = img_name_split[0]
                    if tt == 'Test':
                        img_path_lr = os.path.join(root_lr, f"{img_name}_x1.png")
                        img_path_hr = os.path.join(root_hr, f"{img_name}_{s}.png")
                    else:
                        img_path_lr = os.path.join(root_lr,
                                                   f"{img_name}_x1{img_name_split[1]}")
                        img_path_hr = os.path.join(root_hr,
                                                   f"{img_name}_{s}{img_name_split[1]}")


                    self.read_n -= -1

                    yield {
                        "image_name": img_name,
                        "lr": img_path_lr,
                        "hr": img_path_hr,
                        "scale": s,
                        "is_test": tt == 'Test',
                        "is_train": tt == 'Train'
                    }

    def get_datset_info(self, output_addr='/tmp/drealsdr_info.csv'):

        it = self.get_pair()
        pair_data = next(it)
        columns = list(pair_data.keys())
        columns_str = ",".join(columns)

        with open(output_addr, 'wt') as f:
            f.write(f"{columns_str}\n")


        while True:
            data_str = ",".join([str(pair_data[c]) for c in columns])
            with open(output_addr, 'at') as f:
                f.write(f"{data_str}\n")
            pair_data = next(it)

        return pd.read_csv(output_addr)

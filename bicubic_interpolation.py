from PIL import Image
import tqdm
import os

def bicubic_interp_folder_pil(imgs_folder, scale=4, output_folder=None):

    if output_folder is None:
        output_folder = imgs_folder

    for img_name in tqdm.tqdm((os.listdir(imgs_folder))):
        img_addr =  os.path.join(imgs_folder, img_name)
        im = bicubic_interp(img_addr, scale)
        new_img_name = "".join([img_name.split('.')[0], "x", str(scale), '.png'])
        path_to_save = os.path.join(output_folder, new_img_name)
        im.save(path_to_save)

def bicubic_interp(img_addr, scale=4, shape=None):
    im = Image.open(img_addr)

    if shape is None:
        shape = (int(im.size[0] / scale), int(im.size[1] / scale))

    return im.resize(shape, Image.BICUBIC)

if __name__ == "__main__":
    import sys
    bicubic_interp_folder_pil(sys.argv[1], int(sys.argv[2]), sys.argv[3])


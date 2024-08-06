import argparse
from contextlib import nullcontext
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import numpy as np
import imageio
import PIL, PIL.features, PIL.Image

import torch
# ASYNC
import threading
from multiprocessing.pool import ThreadPool
import shutil
# import pyjion
# pyjion.config(level=2)

class image_iterator:
    def __init__(self, paths, preload_count=2, sync=False):
        self.paths = paths
        self.idx = 0
        self.preload_count = preload_count
        self.thread_list = [
            threading.Thread(target=self.thread_load_image, args=(idx,)) for idx in range(len(self.paths))
        ]
        self.started = [False for idx in range(len(self.paths))]

        self.thread_values = [ None for idx in range(len(self.paths)) ]
        self.sync = sync

        if preload_count > 1:
            if sync:
                self.preload_sync()
            else:
                self.preload()

    def preload(self):
        for idx in range(self.idx, min(self.idx + self.preload_count, len(self.paths))):
            if self.thread_values[idx] is not None or self.thread_list[idx].is_alive() or self.started[idx]:
                continue
            self.started[idx] = True
            self.thread_list[idx].start()

    def thread_load_image(self, idx):
        path = self.paths[idx]
        img = self.preload_image(path)
        self.thread_values[idx] = img

    def preload_sync(self):
        for idx in range(self.idx, min(self.idx + self.preload_count, len(self.paths))):
            if self.thread_values[idx] is not None or self.thread_list[idx].is_alive() or self.started[idx]:
                continue
            self.started[idx] = True
            self.thread_load_image(idx)

    def preload_image(self, path):
        img = None
        if path.endswith('.xml'):
            return None
        if path.endswith('.gif'):
            np_arr =  imageio.mimread(path, memtest=False)[0]
            img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        elif path.endswith('.webp'):
            # Check that it is not an animated webp. If it is extract the first frame.
            try:
                img = PIL.Image.open(path)
            except Exception as e:
                assert PIL.features.check("webp_anim"), "webp_anim not available. Please install webp library (on conda: conda install -c conda-forge libwebp)"
                raise e
            if img.is_animated:
                img.seek(0)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if img is None:
            try:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            except Exception as e:
                print("OpenCV cannot read", path)
        if img is None:
            try:
                img_pil = PIL.Image.open(path)
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print("PIL cannot read", path)
        if img is None:
            try:
                img_imageio = imageio.imread(path)
                img = cv2.cvtColor(img_imageio, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print("imageio cannot read", path)

        if img is None:
            raise Exception("Cannot read", path)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        self.preload()
        if self.idx >= len(self.paths):
            raise StopIteration
        idx = self.idx
        path = self.paths[idx]
        if not self.sync:
            async_img = self.thread_list[idx]
            async_img.join()

        img = self.thread_values[idx]
        self.thread_values[idx] = None

        self.idx += 1

        return idx, path, img

def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-w', '--width', type=int, default=0, help='Output width (ignore scale if set)')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # Print all arguments
    # for arg in vars(args):
    #     print(arg, getattr(args, arg))

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)
    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    # # Enable Tensor cores for FP16
    # if not args.fp32:
    #     torch.backends.cudnn.enabled = True
    # Estimate number of cached images according to their file size
    average_size = 0
    for path in paths:
        average_size += os.path.getsize(path)
    average_size /= len(paths)
    # 500MB of cached images
    size_limit = 500 * 1024 * 1024
    preload_count = int(size_limit / average_size)
    if preload_count < 2:
        preload_count = 2
    # Limit to the number of images
    preload_count = min(preload_count, len(paths))

    print('Preloading', preload_count, 'images')

    img_data_it = image_iterator(paths)
    tasks = []
    for idx, path, img in img_data_it:
        imgname, extension = os.path.splitext(os.path.basename(path))

        # Workaround for cv2.imread() bug with unicode paths
        # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Use of PIL instead of cv2 to read images
        if extension.lower() in {'xml', '.xml'}:
            # Copy xml files
            print('    Copying', idx, imgname + extension)
            shutil.copyfile(path, os.path.join(args.output, imgname + extension))
            continue

        if img is None:
            print('Cannot read', path)
            continue
        print('    Testing', idx + 1, imgname + extension, "shape:", img.shape)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None
        with torch.cuda.amp.autocast() if torch.cuda.is_available() and not args.fp32 else nullcontext():
            try:
                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    scale = args.outscale
                    if args.width != 0:
                        scale = int(round(args.width / img.shape[1]))
                    if scale < 1:
                        scale = 1
                    if scale > netscale:
                        scale = netscale
                    output, _ = upsampler.enhance(img, outscale=scale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                exit(-1)

            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            tasks.append(save_image_async(save_path, output, extension))
    for task in tasks:
        task.join()

SAVE_PARAMS = [
        cv2.IMWRITE_WEBP_QUALITY, 90,
        cv2.IMWRITE_JPEG_QUALITY, 95,
        cv2.IMWRITE_PNG_COMPRESSION, 9,
]

def save_image(save_path, output, extension, params=SAVE_PARAMS):
    # Workaround for cv2.imwrite() bug with unicode paths
    # cv2.imwrite(save_path, output)
    if extension == "webp":
        return save_webp(save_path, output, extension)
    # cv2.imencode('.' + extension, output)[1].tofile(save_path)
    # Same but split images if too large for cv2 (65500 pixels)
    if output.shape[0] > 65500:
        # Keep the same width and split the height
        nb_split = int(np.ceil(output.shape[0] / 65500))
        split_size = int(np.ceil(output.shape[0] / nb_split))
        print("     Warning: image too large for cv2.imwrite(). Splitting into", nb_split, "images.")
        for i in range(nb_split):
            split = output[i * split_size:(i + 1) * split_size, :, :]
            cv2.imencode('.' + extension, split, params)[1].tofile(save_path.replace('.' + extension, f'_{i}.' + extension))
    else:
        cv2.imencode('.' + extension, output, params=params)[1].tofile(save_path)

def save_webp(save_path, output, extension,  params=SAVE_PARAMS):
    # Check if the image is above the max size for webp (16383 x 16383)
    if output.shape[0] > 16383:
        # Keep the same width and split the height
        nb_split = int(np.ceil(output.shape[0] / 16383))
        split_size 	= int(np.ceil(output.shape[0] / nb_split))
        print("     Warning: image too large for cv2.imwrite(). Splitting into", nb_split, "images.")
        for i in range(nb_split):
            split = output[i * split_size:(i + 1) * split_size, :, :]
            cv2.imencode('.' + extension, split, params=params)[1].tofile(save_path.replace('.' 	+ extension, f'_{i}.' 	+ extension))
    else:
        cv2.imencode('.' + extension, output)[1].tofile(save_path)


def save_image_async(save_path, output, extension):
    res = threading.Thread(target=save_image, args=(save_path, output, extension))
    res.start()
    return res

if __name__ == '__main__':
    main()

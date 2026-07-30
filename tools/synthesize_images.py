# Script to add random degradations to images in a folder (offline).
# Mirrors synthesize_datasets.py but works on a flat image directory.

import os
import cv2
import math
import av
import random
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
import torch
from torchvision import transforms
from torch.nn import functional as F


def tensor2img(tensor):
    img = tensor.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# adding gaussian noise
class AddGaussianNoise(object):
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        p = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn(img.size()) * (p / 255.0)
        img = torch.clamp(img + noise, 0.0, 1.0)
        return img


# adding poisson noise
class AddPoissonNoise(object):
    def __init__(self, alpha, beta):  # 2-4
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        alpha = random.uniform(self.alpha, self.beta)
        p = 10 ** alpha
        img = np.random.poisson(img * p) / p
        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img).permute(1, 2, 0)
        img = torch.clamp(img, 0.0, 1.0)
        return img


# adding speckle noise
class AddSpeckleNoise(object):
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        p = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn(img.size()) * (p / 255.0)
        img = img + img * noise
        img = torch.clamp(img, 0.0, 1.0)
        return img


# adding JPEG compression blocking
class AddJPEGCompression(object):
    def __init__(self, comp_level):
        self.comp_level = comp_level

    def __call__(self, img):
        p = random.choice(self.comp_level)
        np_img = img.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        _, encimg = cv2.imencode('.jpg', np_img, [int(cv2.IMWRITE_JPEG_QUALITY), p])
        img = cv2.imdecode(encimg, 3)
        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img)
        return img


# adding video compression artifacts
class AddVideoCompression(object):
    def __init__(self, vcodec):
        self.vcodec = vcodec

    def __call__(self, img):
        vcodec = random.choice(self.vcodec)
        if img.shape[-1] % 2 != 0 or img.shape[-2] % 2 != 0:
            vcodec = 'mpeg4'

        random_seed_suffix = random.randint(0, 100)
        base_root = "./"
        file_name = str(datetime.now()) + "_" + str(random_seed_suffix) + ".mp4"
        file_name = base_root + file_name

        container = av.open(file_name, mode="w")
        stream = container.add_stream(vcodec, rate=1)
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = random.randint(1e4, 1e5)

        img = img.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        stream.width = img.shape[1]
        stream.height = img.shape[0]

        frame = av.VideoFrame.from_ndarray(img)
        packet = stream.encode(frame)
        container.mux(packet)

        packet = stream.encode(None)
        container.mux(packet)
        container.close()

        container = av.open(file_name)
        for t in container.decode(video=0):
            frame = t
        img = frame.to_rgb().to_ndarray()
        container.close()

        os.remove(file_name)
        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img)
        return img


# Gaussian Blur
class AddGaussianBlur(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        kernel_size = random.choice(self.kernel_size)
        kernel = random_mixed_kernels(
            ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            [0.405, 0.225, 0.108, 0.027, 0.108, 0.027],
            kernel_size,
            (0.2, 3),
            (0.2, 3),
            [-math.pi, math.pi],
            noise_range=None,
        )
        img = img.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        img = cv2.filter2D(img, -1, kernel)
        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img)
        return img


class AddResizingBlur(object):
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img):
        h, w = img.shape[-2:]
        updown_type = random.choices(["up", "down", "keep"], [0.3, 0.4, 0.3])[0]
        if updown_type == "up":
            scale = np.random.uniform(1, 2)
        elif updown_type == "down":
            scale = np.random.uniform(0.5, 1)
        else:
            scale = 1

        mode = random.choice(self.mode)

        recover = False
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
            recover = True

        img = F.interpolate(img, scale_factor=scale, mode=mode)
        img = F.interpolate(img, size=(h, w), mode="bicubic")

        img = torch.clamp(img, 0.0, 1.0)
        if recover:
            img = img.squeeze(0)

        return img


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=(0.6, 5),
                         sigma_y_range=(0.6, 5),
                         rotation_range=(-math.pi, math.pi),
                         betag_range=(0.5, 8),
                         betap_range=(0.5, 8),
                         noise_range=None):
    """Randomly generate mixed kernels."""
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        sigma = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        kernel = bivariate_Gaussian(kernel_size, sigma, sigma)
    elif kernel_type == 'aniso':
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, angle, isotropic=False)
    elif kernel_type == 'generalized_iso':
        sigma = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        beta = np.random.uniform(betag_range[0], betag_range[1])
        kernel = bivariate_generalized_Gaussian(kernel_size, sigma, sigma, 0, beta, isotropic=True)
    elif kernel_type == 'generalized_aniso':
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        beta = np.random.uniform(betag_range[0], betag_range[1])
        kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, angle, beta, isotropic=False)
    elif kernel_type == 'plateau_iso':
        sigma = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        beta = np.random.uniform(betap_range[0], betap_range[1])
        kernel = bivariate_plateau(kernel_size, sigma, sigma, 0, beta, isotropic=True)
    elif kernel_type == 'plateau_aniso':
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        beta = np.random.uniform(betap_range[0], betap_range[1])
        kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, angle, beta, isotropic=False)
    else:
        raise ValueError(f'Unsupported kernel type: {kernel_type}')

    return kernel


# --- kernel utilities (copied from synthesize_datasets.py) ---

def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta=0, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta)
    kernel = 1 / (kernel + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_generalized_Gaussian(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel


def sigma_matrix2(sig_x, sig_y, theta):
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return np.dot(np.dot(u_matrix, d_matrix), u_matrix.T)


def mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    grid = np.stack([xx, yy], axis=2)
    return grid, xx, yy


def generateTransforms(prob):
    all_transforms = [
        AddGaussianNoise(10, 15),
        AddPoissonNoise(alpha=2, beta=4),
        AddSpeckleNoise(10, 15),
        AddJPEGCompression([20, 30, 40]),
        AddVideoCompression(['libx264', 'h264', 'mpeg4']),
        AddGaussianBlur([3, 5, 7]),
        AddResizingBlur(["area", "bilinear", "bicubic"]),
    ]
    random.shuffle(all_transforms)
    selected_transforms = [t for t in all_transforms if random.random() > prob]
    return transforms.Compose(selected_transforms)


def is_image_file(name):
    name = name.lower()
    return name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.bmp')


def main(input_dir, output_dir, prob=0.55):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    toTensor = transforms.Compose([transforms.ToTensor()])

    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if not is_image_file(name):
                continue

            src_path = os.path.join(root, name)
            rel_path = os.path.relpath(src_path, input_dir)
            dst_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            img = Image.open(src_path).convert('RGB')
            img = toTensor(img)

            deg_transform = generateTransforms(prob)
            img = deg_transform(img)
            img = tensor2img(img)

            cv2.imwrite(dst_path, img)

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add random degradations to images in a folder.")
    parser.add_argument('--inp', type=str, help='Input directory containing images.')
    parser.add_argument('--out', type=str, help='Output directory for degraded images.')
    parser.add_argument('--input_dir', type=str, help='Input directory containing images. (alias of --inp)')
    parser.add_argument('--output_dir', type=str, help='Output directory for degraded images. (alias of --out)')
    parser.add_argument('--prob', type=float, default=0.55, help='Probability to skip a transformation.')

    args = parser.parse_args()
    input_dir = args.inp or args.input_dir
    output_dir = args.out or args.output_dir
    if not input_dir or not output_dir:
        parser.error("Both input and output directories are required. Use --inp/--out (or --input_dir/--output_dir).")
    main(input_dir, output_dir, args.prob)

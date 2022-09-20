import math

import torch
from skimage import io
from torchvision.transforms import Resize


def create_gaussian_pyramid(img, scale_factor, num_levels):
    reals = []
    stop_scale = num_levels - 1
    h, w = img.shape[-2:]
    for i in range(stop_scale):
        scale = scale_factor ** (stop_scale - i)
        curr_real = Resize((int(h*scale), int(w*scale)), antialias=True)(img)
        reals.append(curr_real)
    reals.append(img)
    return reals


def read_image(path):
    x = io.imread(path)
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1).unsqueeze(0).float()
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 2 - 1
    return x


def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def build_reference_pyramid(image_paths, resize, coarse_dim, num_levels, device):
    # Define reference image pyramid
    reference_images = [read_image(image_path).to(device) for image_path in image_paths]
    if resize is not None:
        reference_images = [Resize(resize, antialias=True)(img) for img in reference_images]
    reference_images = torch.cat(reference_images, dim=0)
    min_dim = min(reference_images.shape[2], reference_images.shape[3])
    scale_factor = math.pow(coarse_dim / min_dim, 1 / (num_levels - 1))
    reference_pyramid = create_gaussian_pyramid(reference_images, scale_factor, num_levels)
    return reference_pyramid

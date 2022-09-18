import os
import math
from copy import deepcopy
import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Resize
from tqdm import tqdm

from argparse import Namespace
from models import weights_init, Generator, reset_grads, ArrayOFGenerators
from utils import read_image, create_gaussian_pyramid


def get_models(opt):
    netG = Generator(opt).to(device)
    netG.apply(weights_init)
    netG.train()
    return netG


class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256):
        super(PatchSWDLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj

    def forward(self, x, y):
        b, c, h, w = x.shape

        # Sample random normalized projections
        rand = torch.randn(self.num_proj, c*self.patch_size**2).to(x.device) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        rand = rand.reshape(self.num_proj, c, self.patch_size, self.patch_size)

        # Project patches
        projx = F.conv2d(x, rand).transpose(1,0).reshape(self.num_proj, -1)
        projy = F.conv2d(y, rand).transpose(1,0).reshape(self.num_proj, -1)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(projx - projy).mean()

        return loss


def main(image_path, opt):
    # Define reference image pyramid
    reference_image = read_image(image_path).to(device)
    min_dim = min(reference_image.shape[2], reference_image.shape[3])
    scale_factor = math.pow(opt.min_size / min_dim, 1 / (opt.num_levels - 1))
    reference_pyr = create_gaussian_pyramid(reference_image, scale_factor, opt.num_levels)

    fixed_zs = []
    multi_scale_generator = ArrayOFGenerators()
    for lvl in range(opt.num_levels):

        if lvl % opt.reinit_freq == 0: #
            netG = get_models(opt)

        save_image(reference_pyr[lvl], f"{output_dir}/Reference_lvl-{lvl}.png", normalize=True)

        netG, shape, noise_amp, cur_lvl_fixed_z = train_single_scale(multi_scale_generator, netG, reference_pyr, fixed_zs, opt)
        fixed_zs.append(cur_lvl_fixed_z)

        # Freeze model before adding it into the array of previous models
        G_curr = reset_grads(deepcopy(netG), False)
        # G_curr.eval()
        multi_scale_generator.append(G_curr, shape, noise_amp)


def train_single_scale(multi_scale_generator, netG, reference_pyr, prev_fixed_zs, opt):
    batch_size = 16
    lvl = len(multi_scale_generator)
    ref_img = reference_pyr[lvl].to(device).repeat(batch_size, 1, 1, 1)
    cur_h, cur_w = ref_img.shape[-2:]
    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[opt.niter],gamma=opt.gamma)

    rec_criteria = nn.MSELoss()
    criteria = PatchSWDLoss(patch_size=16, num_proj=256)
    if lvl == 0:
        noise_amp = 1
    else:
        with torch.no_grad():
            fixed_prev = multi_scale_generator.sample_images(prev_fixed_zs)
            RMSE = torch.sqrt(rec_criteria(fixed_prev, reference_pyr[lvl -1]))
            noise_amp = opt.base_noise_amp * RMSE

    cur_lvl_fixed_z = torch.randn(1, 3, cur_h, cur_w, device=device) * noise_amp

    print(f"Lvl- {lvl}: shape: {(cur_h, cur_w)}. noise_amp: {noise_amp}")
    for iter in tqdm(range(opt.niter)):
        ############################
        # (2) Update G network: maximize D(G(z))
        #########################
        netG.zero_grad()
        prev = 0
        if lvl > 0:
            prev = multi_scale_generator.sample_images(multi_scale_generator.sample_zs(batch_size))
            prev = Resize((cur_h, cur_w), antialias=True)(prev)

        noise = torch.randn(batch_size, 3, cur_h, cur_w, device=device) * noise_amp

        fake = netG(prev + noise) + prev
        errG = criteria(fake, ref_img)

        if lvl == 0:
            fixed_prev = torch.zeros_like(cur_lvl_fixed_z)
        else:
            fixed_prev = multi_scale_generator.sample_images(prev_fixed_zs)
            fixed_prev = Resize((cur_h, cur_w), antialias=True)(fixed_prev)

        rec = netG(fixed_prev + cur_lvl_fixed_z) + fixed_prev
        rec_loss = rec_criteria(rec, ref_img[:1])

        errG_total = errG + cfg.rec_weight * rec_loss
        errG_total.backward()

        optimizerG.step()

        schedulerG.step()

        ############################
        # (3) Log Results
        ###########################
        if iter % 200 == 0 or iter+1 == opt.niter:
            with torch.no_grad():
                netG.eval()
                fake = (netG(prev + noise) + prev).clip(-1,1)
                rec = (netG(fixed_prev + cur_lvl_fixed_z) + fixed_prev).clip(-1, 1)  # Zero noise
                netG.train()
            save_image(fake, f"{output_dir}/Fake_lvl-{lvl}-iter-{iter}.png", normalize=True)
            save_image(rec, f"{output_dir}/Reconstruction_lvl-{lvl}-iter-{iter}.png", normalize=True)

    return netG, (cur_h, cur_w), noise_amp, cur_lvl_fixed_z


if __name__ == '__main__':
    device = torch.device("cuda:0")

    cfg = Namespace()
    cfg.num_levels = 8              # Number of scales in the image pyramid
    cfg.min_size = 25               # Dimentsion of the coarsest level in the pyramid
    cfg.nfc = 32                    # number of convolution channels in each block
    cfg.num_model_blocks = 5        # How many convolution blocks in each Generator / Discriminator
    cfg.niter = 1000                # Number of gradient steps at each level
    cfg.lr = 0.0005                 # Adam learning rate
    cfg.base_noise_amp = 0.1        # noise amplitude for levels > 0 are multiplied by this factor
    cfg.gp_weight = 0.1             # Gradient penalty weight in total loss
    cfg.rec_weight = 10             # L2 reconstruction weight in total loss
    cfg.reinit_freq = 4             # How frequently (scales) to reset the generator & critic weights.
    cfg.gamma = 0.1                 # LR schedule gamma parameter

    output_dir = f"Outputs-SWD"
    os.makedirs(output_dir, exist_ok=True)
    main("Images/balloons.png", cfg)
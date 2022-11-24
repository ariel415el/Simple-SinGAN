import os
import math
from copy import deepcopy
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Resize
from tqdm import tqdm

from argparse import Namespace

from utils import build_reference_pyramid
from models import weights_init, Generator, reset_grads, ArrayOFGenerators


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


def main(image_paths, opt):
    reference_pyramid = build_reference_pyramid(image_paths, opt.resize, opt.coarse_dim, opt.num_levels, device)
    nrow = int(math.sqrt(len(image_paths)))

    multi_scale_generator = ArrayOFGenerators()
    fixed_zs = []
    all_losses = []
    for lvl in range(opt.num_levels):

        if lvl % opt.models_reset_freq == 0: #
            netG = get_models(opt)

        save_image(reference_pyramid[lvl], f"{output_dir}/Reference_lvl-{lvl}.png", normalize=True, nrow=nrow)

        netG, shape, noise_amp, cur_lvl_fixed_z, lvl_losses = train_single_scale(multi_scale_generator, netG, reference_pyramid, fixed_zs, opt)
        fixed_zs.append(cur_lvl_fixed_z)

        # Freeze model before adding it into the array of previous models
        G_curr = reset_grads(deepcopy(netG), False)
        G_curr.eval()
        multi_scale_generator.append(G_curr, shape, noise_amp)

        images = multi_scale_generator.sample_images(multi_scale_generator.sample_zs(16))
        save_image(images.clip(-1, 1), os.path.join(output_dir, f"Val-Samples-{lvl}.png"), normalize=True, nrow=nrow)
        all_losses += lvl_losses
        plt.plot(range(len(all_losses)), all_losses)
        plt.savefig(os.path.join(output_dir, "Losses.png"))

def train_single_scale(multi_scale_generator, netG, reference_pyr, prev_fixed_zs, opt):
    lvl = len(multi_scale_generator)
    reference_image = reference_pyr[lvl].to(device)
    batch_size = len(reference_image)
    cur_h, cur_w = reference_image.shape[-2:]

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[opt.niter],gamma=opt.gamma)

    rec_criteria = nn.MSELoss()

    if lvl == 0:
        noise_amp = 1
    else:
        with torch.no_grad():
            fixed_prev = multi_scale_generator.sample_images(prev_fixed_zs)
            RMSE = torch.sqrt(rec_criteria(fixed_prev, reference_pyr[lvl -1]))
            noise_amp = opt.base_noise_amp * RMSE

    cur_lvl_fixed_z = torch.randn(1, 3, cur_h, cur_w, device=device) * noise_amp

    print(f"Lvl- {lvl}: shape: {(cur_h, cur_w)}. noise_amp: {noise_amp}")
    losses = []
    for iter in tqdm(range(opt.niter)):
        ############################
        # (1) Update G network: maximize D(G(z))
        #########################
        netG.zero_grad()
        prev = 0
        if lvl > 0:
            prev = multi_scale_generator.sample_images(multi_scale_generator.sample_zs(batch_size))
            prev = Resize((cur_h, cur_w), antialias=True)(prev)

        noise = torch.randn(batch_size, 3, cur_h, cur_w, device=device) * noise_amp
        fake = netG(prev + noise) + prev

        ref_img = reference_image.clone()

        errG = criteria(fake, ref_img)

        if lvl == 0:
            fixed_prev = torch.zeros_like(cur_lvl_fixed_z)
        else:
            fixed_prev = multi_scale_generator.sample_images(prev_fixed_zs)
            fixed_prev = Resize((cur_h, cur_w), antialias=True)(fixed_prev)

        reconstruction = netG(fixed_prev + cur_lvl_fixed_z) + fixed_prev
        rec_loss = rec_criteria(reconstruction, ref_img[:1])

        errG_total = errG + cfg.rec_weight * rec_loss

        errG_total.backward()
        optimizerG.step()
        schedulerG.step()

        losses.append(errG.item())

        ############################
        # (3) Log Results
        ###########################
        if iter % 1000 == 0 or iter+1 == opt.niter:
            nrow = int(math.sqrt(len(fake)))
            save_image(fake.clip(-1,1), os.path.join(output_dir, f"Train-samples_lvl-{lvl}-iter-{iter}.png"), normalize=True, nrow=nrow)
            save_image(reconstruction.clip(-1, 1), os.path.join(output_dir, f"Train-reconstruction_lvl-{lvl}-iter-{iter}.png"), normalize=True)

    return netG, (cur_h, cur_w), noise_amp, cur_lvl_fixed_z, losses


if __name__ == '__main__':
    device = torch.device("cuda:0")

    cfg = Namespace()
    cfg.resize = None               # Resize images to this size
    cfg.num_levels = 7              # Number of scales in the image pyramid
    cfg.coarse_dim = 12               # Dimentsion of the coarsest level in the pyramid
    cfg.nfc = 32                    # number of convolution channels in each block
    cfg.num_model_blocks = 5        # How many convolution blocks in each Generator / Discriminator
    cfg.niter = 10000                # Number of gradient steps at each level
    cfg.lr = 0.001                  # Adam learning rate
    cfg.base_noise_amp = 0.1        # noise amplitude for levels > 0 are multiplied by this factor
    cfg.gp_weight = 0.1             # Gradient penalty weight in total loss
    cfg.rec_weight = 10             # L2 reconstruction weight in total loss
    cfg.models_reset_freq = 4             # How frequently (scales) to reset the generator & critic weights.
    cfg.gamma = 0.1                 # LR schedule gamma parameter

    criteria = PatchSWDLoss(patch_size=7, num_proj=16)
    output_dir = f"Outputs-SWD"
    os.makedirs(output_dir, exist_ok=True)
    main(["Images/balloons.png"] * 16, cfg)
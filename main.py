import os
from copy import deepcopy
import torch
from torch import optim, nn
from torchvision.utils import save_image
from torchvision.transforms import Resize
from tqdm import tqdm

from argparse import Namespace

from diffaug import DiffAugment
from main_SWD import PatchSWDLoss
from models import weights_init, Generator, WDiscriminator, reset_grads, ArrayOFGenerators
from utils import calc_gradient_penalty, build_reference_pyramid, plot_losses


def get_models(opt):
    netG = Generator(opt).to(device)
    netG.apply(weights_init)
    netG.train()
    netD = WDiscriminator(opt).to(device)
    netD.train()
    netD.apply(weights_init)
    return netG, netD


def main(image_paths, opt):
    reference_pyramid = build_reference_pyramid(image_paths, opt.resize, opt.coarse_dim, opt.num_levels, device)

    swd_losses = []
    fixed_zs = []
    multi_scale_generator = ArrayOFGenerators()
    for lvl in range(opt.num_levels):

        if lvl % opt.models_reset_freq == 0:
            netG, netD = get_models(opt)

        save_image(reference_pyramid[lvl], f"{output_dir}/Reference_lvl-{lvl}.png", normalize=True)

        netG, shape, noise_amp, cur_lvl_fixed_z, losses = train_single_scale(multi_scale_generator, netG, netD, reference_pyramid, fixed_zs, opt)
        fixed_zs.append(cur_lvl_fixed_z)

        # Freeze model before adding it into the array of previous models
        G_curr = reset_grads(deepcopy(netG), False)
        G_curr.eval()
        multi_scale_generator.append(G_curr, shape, noise_amp)

        # Dump samples
        images = multi_scale_generator.sample_images(multi_scale_generator.sample_zs(16))
        save_image(images.clip(-1, 1), os.path.join(output_dir, f"Val-Samples-{lvl}.png"), normalize=True, nrow=4)
        swd_losses += losses
        plot_losses(swd_losses, os.path.join(output_dir, "Losses.png"))


def train_single_scale(multi_scale_generator, netG, netD, reference_pyr, fixed_previous_zs, opt):
    lvl = len(multi_scale_generator)
    reference_image = reference_pyr[lvl].to(device)
    batch_size = len(reference_image)
    cur_h, cur_w = reference_image.shape[-2:]

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[opt.niter],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[opt.niter],gamma=opt.gamma)

    rec_criteria = nn.MSELoss()
    if lvl == 0:
        noise_amp = 1
    else:
        with torch.no_grad():
            fixed_previous_image = multi_scale_generator.sample_images(fixed_previous_zs)
            RMSE = torch.sqrt(rec_criteria(fixed_previous_image, reference_pyr[lvl -1]))
            noise_amp = opt.base_noise_amp * RMSE

    cur_lvl_fixed_z = torch.randn(batch_size, 3, cur_h, cur_w, device=device) * noise_amp

    print(f"Lvl- {lvl}: shape: {(cur_h, cur_w)}. noise_amp: {noise_amp:.3f}")
    losses = []
    for iter in tqdm(range(opt.niter)):
        # Draw input for this level generator from previous scales
        previous_image = 0
        if lvl > 0:
            previous_zs = multi_scale_generator.sample_zs(batch_size)
            previous_image = multi_scale_generator.sample_images(previous_zs)
            previous_image = Resize((cur_h, cur_w), antialias=True)(previous_image)
        noise = torch.randn(batch_size, 3, cur_h, cur_w, device=device) * noise_amp
        fake = netG(previous_image + noise) + previous_image

        ref_img = reference_image.clone()
        if opt.augment:
            ref_img = DiffAugment(ref_img, prob=iter/opt.niter, policy=opt.augment)
            fake = DiffAugment(fake, prob=iter/opt.niter, policy=opt.augment)

        ############################
        # (1) Update D network: maximize D(G(z))
        #########################
        netD.zero_grad()
        errD_real = -netD(ref_img).mean()
        errD_fake = netD(fake.detach()).mean()
        gradient_penalty = calc_gradient_penalty(netD, ref_img, fake, device)
        errD_total = errD_real + errD_fake + opt.gp_weight * gradient_penalty

        errD_total.backward()
        optimizerD.step()
        schedulerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        #########################
        netG.zero_grad()
        errG = -netD(fake).mean()

        if lvl == 0:
            fixed_previous_image = torch.zeros_like(cur_lvl_fixed_z)
        else:
            fixed_previous_image = multi_scale_generator.sample_images(fixed_previous_zs)
            fixed_previous_image = Resize((cur_h, cur_w), antialias=True)(fixed_previous_image)

        reconstruction = netG(fixed_previous_image + cur_lvl_fixed_z) + fixed_previous_image
        rec_loss = rec_criteria(reconstruction, reference_image)

        errG_total = errG + cfg.rec_weight * rec_loss
        errG_total.backward()

        optimizerG.step()
        schedulerG.step()

        with torch.no_grad():
            losses.append(debug_criteria(ref_img, fake).item())

        ###########################
        # (3) Log Results
        ############################
        if iter % 1000 == 0 or iter+1 == opt.niter:
            save_image(fake.clip(-1,1), os.path.join(output_dir, f"Train-samples_lvl-{lvl}-iter-{iter}.png"), normalize=True)
            save_image(reconstruction.clip(-1, 1), os.path.join(output_dir, f"Train-reconstruction_lvl-{lvl}-iter-{iter}.png"), normalize=True)

    return netG, (cur_h, cur_w), noise_amp, cur_lvl_fixed_z, losses


if __name__ == '__main__':
    device = torch.device("cuda:0")

    cfg = Namespace()
    cfg.resize = None               # Resize images to this size
    cfg.num_levels = 8                # Number of scales in the image pyramid
    cfg.coarse_dim = 25                 # Dimentsion of the coarsest level in the pyramid
    cfg.nfc = 32                      # number of convolution channels in each block
    cfg.num_model_blocks = 6          # How many convolution blocks in each Generator / Discriminator
    cfg.niter = 4000                  # Number of gradient steps at each level
    cfg.lr = 0.0005                   # Adam learning rate
    cfg.base_noise_amp = 0.1          # noise amplitude for levels > 0 are multiplied by this factor
    cfg.gp_weight = 0.1               # Gradient penalty weight in total loss
    cfg.rec_weight = 10               # L2 reconstruction weight in total loss
    cfg.models_reset_freq = 100         # How frequently (scales) to reset the generator & critic weights.
    cfg.gamma = 0.1                   # LR schedule gamma parameter
    cfg.augment = '' # Data augmentation

    debug_criteria = PatchSWDLoss(patch_size=7, num_proj=64)

    output_dir = f"Outputs"
    os.makedirs(output_dir, exist_ok=True)
    main(["Images/balloons.png"], cfg)
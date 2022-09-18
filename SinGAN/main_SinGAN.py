import os
import math
from copy import deepcopy
import torch
from torch import optim, nn
from torchvision.utils import save_image
from torchvision.transforms import Resize
from tqdm import tqdm

from argparse import Namespace
from SinGAN.models import weights_init, Generator, WDiscriminator, reset_grads, ArrayOFGenerators
from utils import read_image, create_gaussian_pyramid, calc_gradient_penalty


def get_models(opt):
    netG = Generator(opt).to(device)
    netG.apply(weights_init)
    netG.train()
    netD = WDiscriminator(opt).to(device)
    netD.train()
    netD.apply(weights_init)
    return netG, netD


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
            netG, netD = get_models(opt)

        save_image(reference_pyr[lvl], f"{output_dir}/Reference_lvl-{lvl}.png", normalize=True)

        netG, shape, noise_amp, cur_lvl_fixed_z = train_single_scale(multi_scale_generator, netG, netD, reference_pyr, fixed_zs, opt)
        fixed_zs.append(cur_lvl_fixed_z)

        # Freeze model before adding it into the array of previous models
        G_curr = reset_grads(deepcopy(netG), False)
        # G_curr.eval()
        multi_scale_generator.append(G_curr, shape, noise_amp)


def train_single_scale(multi_scale_generator, netG, netD, reference_pyr, prev_fixed_zs, opt):
    lvl = len(multi_scale_generator)
    ref_img = reference_pyr[lvl].to(device)
    cur_h, cur_w = ref_img.shape[-2:]
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[opt.niter],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[opt.niter],gamma=opt.gamma)

    rec_criteria = nn.MSELoss()
    if lvl == 0:
        noise_amp = 1
    else:
        with torch.no_grad():
            fixed_prev = multi_scale_generator.draw(prev_fixed_zs)
            RMSE = torch.sqrt(rec_criteria(fixed_prev, reference_pyr[lvl -1]))
            noise_amp = opt.base_noise_amp * RMSE

    cur_lvl_fixed_z = torch.randn(1, 3, cur_h, cur_w, device=device) * noise_amp

    print(f"Lvl- {lvl}: shape: {(cur_h, cur_w)}. noise_amp: {noise_amp}")
    for iter in tqdm(range(opt.niter)):
        ############################
        # (1) Update D network: maximize D(ref_img) - D(G(z))
        ###########################
        # train with real
        netD.zero_grad()
        errD_real = -netD(ref_img).mean()

        # Draw input for this level generator from previous scales  
        prev = 0
        if lvl > 0:
            prev = multi_scale_generator.draw()
            prev = Resize((cur_h, cur_w), antialias=True)(prev)

        # train with fake
        noise = torch.randn(1, 3, cur_h, cur_w, device=device) * noise_amp
        fake = netG(prev + noise) + prev

        errD_fake = netD(fake.detach()).mean()

        gradient_penalty = calc_gradient_penalty(netD, ref_img, fake, device)

        errD_total = errD_real + errD_fake + opt.gp_weight * gradient_penalty
        errD_total.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        #########################
        netG.zero_grad()
        errG = -netD(fake).mean()

        if lvl == 0:
            fixed_prev = torch.zeros_like(cur_lvl_fixed_z)
        else:
            fixed_prev = multi_scale_generator.draw(prev_fixed_zs)
            fixed_prev = Resize((cur_h, cur_w), antialias=True)(fixed_prev)

        rec = netG(fixed_prev + cur_lvl_fixed_z) + fixed_prev
        rec_loss = rec_criteria(rec, ref_img)

        errG_total = errG + cfg.rec_weight * rec_loss
        errG_total.backward()

        optimizerG.step()

        schedulerD.step()
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
    cfg.min_size = 32               # Dimentsion of the coarsest level in the pyramid
    cfg.nfc = 32                    # number of convolution channels in each block
    cfg.num_model_blocks = 5        # How many convolution blocks in each Generator / Discriminator
    cfg.niter = 4000                # Number of gradient steps at each level
    cfg.lr = 0.0005                 # Adam learning rate
    cfg.base_noise_amp = 0.1        # noise amplitude for levels > 0 are multiplied by this factor
    cfg.gp_weight = 0.1             # Gradient penalty weight in total loss
    cfg.rec_weight = 10             # L2 reconstruction weight in total loss
    cfg.reinit_freq = 4             # How frequently (scales) to reset the generator & critic weights.
    cfg.gamma = 0.1                 # LR schedule gamma parameter

    output_dir = f"Outputs-"
    os.makedirs(output_dir, exist_ok=True)
    main("../balloons.png", cfg)
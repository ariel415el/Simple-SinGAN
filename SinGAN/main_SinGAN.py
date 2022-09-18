import os
import math
from copy import deepcopy
import torch
from torch import optim, nn
from torchvision.utils import save_image
from torchvision.transforms import Resize
from tqdm import tqdm

from argparse import Namespace
from SinGAN.models import weights_init, GeneratorConcatSkip2CleanAdd, WDiscriminator
from utils import read_image_dir, create_reals_pyramid, calc_gradient_penalty

torch.autograd.set_detect_anomaly(True)

def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

class ArrayOFGenerators:
    def __init__(self):
        self.Gs = []
        self.shapes = []
        self.noise_amps = []

    def append(self, netG, shape, noise_amp):
        self.Gs.append(netG)
        self.shapes.append(shape)
        self.noise_amps.append(noise_amp)

    def draw(self, zs=None):
        if not self.Gs:
            return 0

        if zs is None:
            zs = []
            for shape, amp in zip(self.shapes, self.noise_amps):
                zs += [torch.randn(1, 3, shape[0], shape[1], device=device) * amp]

        output = 0
        for i, (z, G) in enumerate(zip(zs, self.Gs)):
            output = G(z + output) + output
            if i != len(self.Gs) - 1:
                output = Resize(zs[i+1].shape[-2:], antialias=True)(output)
        return output

    def __len__(self):
        return len(self.Gs)


def get_models(opt):
    netG = GeneratorConcatSkip2CleanAdd(opt).to(device)
    netG.apply(weights_init)
    netG.train()
    netD = WDiscriminator(opt).to(device)
    netD.train()
    netD.apply(weights_init)
    return netG, netD


def main(image_path, opt):
    reference_image = read_image_dir(image_path).to(device)
    min_dim = min(reference_image.shape[2], reference_image.shape[3])
    scale_factor = math.pow(opt.min_size / min_dim, 1 / (opt.num_levels - 1))
    reference_pyr = create_reals_pyramid(reference_image, scale_factor, opt)

    multi_scale_generator = ArrayOFGenerators()
    fixed_noise = torch.randn(1, 3, reference_pyr[0].shape[2], reference_pyr[0].shape[3], device=device)

    for lvl in range(opt.num_levels):

        if lvl % 4 == 0:
            netG, netD = get_models(opt)

        save_image(reference_pyr[lvl], f"{output_dir}/Reference_lvl-{lvl}.png", normalize=True)

        netG, shape, noise_amp = train_single_scale(multi_scale_generator, netG, netD, reference_pyr, fixed_noise, opt)

        G_curr = reset_grads(deepcopy(netG), False)
        # G_curr.eval()
        multi_scale_generator.append(G_curr, shape, noise_amp)


def train_single_scale(multi_scale_generator, netG, netD, reals, lvl0_fixed_noise, opt):
    lvl = len(multi_scale_generator)
    shapes = [real.shape for real in reals[:lvl]]
    real = reals[lvl].to(device)
    cur_h, cur_w = real.shape[-2:]
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[opt.niter],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[opt.niter],gamma=opt.gamma)

    rec_criteria = nn.MSELoss()
    if lvl == 0:
        prev_fixed_zs = []
        cur_lvl_fixed_z = lvl0_fixed_noise
        noise_amp = 1
    else:
        prev_fixed_zs = [lvl0_fixed_noise] + [torch.zeros(*shape, device=device) for shape in shapes[1:]]
        cur_lvl_fixed_z = torch.zeros(1, 3, cur_h, cur_w, device=device)
        with torch.no_grad():
            fixed_prev = multi_scale_generator.draw(prev_fixed_zs)
            RMSE = torch.sqrt(rec_criteria(fixed_prev, reals[lvl -1]))
            noise_amp = opt.base_noise_amp * RMSE
    print(f"Lvl- {lvl}: shape: {(cur_h, cur_w)}. noise_amp: {noise_amp}")
    for iter in tqdm(range(opt.niter)):
        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        # train with real
        netD.zero_grad()
        output = netD(real)
        errD_real = -output.mean()

        # train with fake
        noise = torch.randn(1, 3, cur_h, cur_w, device=device) * noise_amp
        prev = 0
        if lvl > 0:
            prev = multi_scale_generator.draw()
            prev = Resize((cur_h, cur_w), antialias=True)(prev)

        fake = netG(noise + prev) + prev

        output = netD(fake.detach())
        errD_fake = output.mean()

        gradient_penalty = calc_gradient_penalty(netD, real, fake, device)

        errD_total = errD_real + errD_fake + opt.lambda_grad * gradient_penalty
        errD_total.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        #########################
        netG.zero_grad()
        output = netD(fake)
        errG = -output.mean()

        if lvl == 0:
            fixed_prev = torch.zeros_like(cur_lvl_fixed_z)
        else:
            fixed_prev = multi_scale_generator.draw(prev_fixed_zs)
            fixed_prev = Resize((cur_h, cur_w), antialias=True)(fixed_prev)

        rec = netG(fixed_prev + cur_lvl_fixed_z) + fixed_prev  # Zero noise
        rec_loss = rec_criteria(rec, real)

        errG_total = errG + cfg.rec_coeff * rec_loss
        errG_total.backward()

        optimizerG.step()

        schedulerD.step()
        schedulerG.step()

        ############################
        # (3) Log Results
        ###########################
        if iter % 1000 == 0 or iter+1 == opt.niter:
            with torch.no_grad():
                netG.eval()
                fake = (netG(noise + prev) + prev).clip(-1,1)
                rec = (netG(fixed_prev + cur_lvl_fixed_z) + fixed_prev).clip(-1, 1)  # Zero noise
                netG.train()
            save_image(fake, f"{output_dir}/Fake_lvl-{lvl}-iter-{iter}.png", normalize=True)
            save_image(rec, f"{output_dir}/Reconstruction_lvl-{lvl}-iter-{iter}.png", normalize=True)

    return netG, (cur_h, cur_w), noise_amp


if __name__ == '__main__':
    device = torch.device("cuda:0")
    output_dir = "outputs"


    cfg = Namespace()
    cfg.min_size = 25
    cfg.num_levels = 8
    cfg.nfc = 32
    cfg.num_layer = 5
    cfg.lr = 0.0005
    cfg.base_noise_amp = 0.1
    cfg.lambda_grad = 0.1
    cfg.niter = 4000
    cfg.rec_coeff = 10
    cfg.gamma = 0.1

    os.makedirs(output_dir, exist_ok=True)
    main("balloons.png", cfg)
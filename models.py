import torch
import torch.nn as nn
from torchvision.transforms import Resize


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=False))


class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        ker_size = 3
        padd_size = 0
        N = int(opt.nfc)
        self.head = ConvBlock(3, N, ker_size, padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_model_blocks - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.nfc), max(N, opt.nfc), ker_size, padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.nfc), 1, kernel_size=ker_size, stride=1, padding=padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        ker_size = 3
        # padd_size = 1
        padd_size = 0
        self.padding = (ker_size // 2) * opt.num_model_blocks

        N = opt.nfc
        self.head = ConvBlock(3, N, ker_size, padd_size, 1)

        self.body = nn.Sequential()
        for i in range(opt.num_model_blocks - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.nfc), max(N, opt.nfc), ker_size, padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.nfc), 3, kernel_size=ker_size, stride=1, padding=padd_size),
            nn.Tanh()
        )

    def forward(self, x):
        import torch.nn.functional as F
        x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x


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

    def sample_zs(self, n_samples):
        """Sample spacial noise in the correct shapes for the trained generators and with the right amplitudes"""
        device = next(self.Gs[0].parameters()).device
        zs = []
        for shape, amp in zip(self.shapes, self.noise_amps):
            zs += [torch.randn(n_samples, 3, shape[0], shape[1], device=device) * amp]
        return zs

    def sample_images(self, zs=None):
        """
        Gradualy synthesize an image by running the generators on the supplied spacial latents.
         If zs are not supplied they are sampled.
         """
        if not self.Gs:
            return 0

        if zs is None:
            zs = self.sample_zs(1)

        output = 0
        for i, (z, G) in enumerate(zip(zs, self.Gs)):
            output = G(z + output) + output
            if i != len(self.Gs) - 1:
                output = Resize(zs[i+1].shape[-2:], antialias=True)(output)
        return output

    def __len__(self):
        return len(self.Gs)
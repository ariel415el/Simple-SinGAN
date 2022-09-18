import torch
import torch.nn as nn

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
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.nfc), max(N, opt.nfc), ker_size, padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.nfc), 1, kernel_size=ker_size, stride=1, padding=padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        ker_size = 3
        # padd_size = 1
        padd_size = 0
        self.padding = (ker_size // 2) * opt.num_layer

        N = opt.nfc
        self.head = ConvBlock(3, N, ker_size, padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
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

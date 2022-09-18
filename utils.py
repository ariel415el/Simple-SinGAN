import torch
from skimage import io
from torchvision.transforms import Resize


def create_reals_pyramid(real, scale_factor, opt):
    reals = []
    stop_scale = opt.num_levels - 1
    h, w = real.shape[-2:]
    for i in range(stop_scale):
        # scale = math.pow(scale_factor,((stop_scale-1)/math.log(stop_scale))*math.log(stop_scale-i)+1)
        scale = scale_factor ** (stop_scale - i)
        curr_real = Resize((int(h*scale), int(w*scale)), antialias=True)(real)
        reals.append(curr_real)
    reals.append(real)
    return reals


def read_image_dir(path):
    x = io.imread(path)
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1).unsqueeze(0).float()
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 2 - 1
    # from PIL import Image
    # import torchvision
    # x = Image.open(path).convert('RGB')
    # x = torchvision.transforms.ToTensor()(x).unsqueeze(dim=0)
    # x = (x - 0.5) * 2
    # x.clamp(-1, 1)
    return x



def calc_gradient_penalty(netD, real_data, fake_data, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

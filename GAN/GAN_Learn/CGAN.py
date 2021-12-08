from __future__ import print_function

# %matplotlib inline
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset 数据集
dataroot = "data/celeba/img_align_celeba"

# Number of workers for dataloader
workers = 2

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 32

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 20

# Size of feature maps in generator
ngf = 32

# Size of feature maps in discriminator
ndf = 32

# Number of training epochs
num_epochs = 16

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

condition_dim = 10

##########################################################加载数据######################################################
# We can use an image folder dataset the way we have it setup.
# Create the dataset

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def loadData():
    dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(image_size),
                                             transforms.CenterCrop(image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5), (0.5)),
                                         ])
                                         )

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                             shuffle=True, num_workers=workers)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    return dataloader


# custom weights initialization called on netG and netD
def weights_init(m):  # 模型的权重进行初始化
    classname = m.__class__.__name__  # 类名称
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
# 生成器
class Generator(nn.Module):
    def __init__(self, condition_dim):
        super(Generator, self).__init__()
        self.condition_dim = condition_dim
        first_c = 258
        second_c = 128
        third_c = 64
        self.condition_embedding = nn.Embedding(condition_dim, condition_dim)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # 第一个参数表示输入通道的数量
            # 第二个表示输出通道的数量
            # 注意这里的实现和论文图片有一定的区别，没有第二层reshape成1024那一段
            # 另外不要竖着看最开始那个变量，横着看，它的大小是1*1*100，通道数100，不是长宽100*1
            nn.ConvTranspose2d(nz + condition_dim, first_c, 4, 1, 0, bias=False),
            nn.BatchNorm2d(first_c),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(first_c, second_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(second_c),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(second_c, third_c, 4, 2, 1, bias=False),  # 16*16*128
            nn.BatchNorm2d(third_c),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(third_c, nc, 4, 2, 1, bias=False),  # 32 * 32* 1
            nn.Tanh()  # tanh变到-1-1之间
            # state size. (nc) x 64 x 64
            # 最后的输出图片 通道数*64*64
        )

    def forward(self, input, condition):
        condition = self.condition_embedding(condition)
        condition = condition.unsqueeze(2).unsqueeze(3)
        return self.main(torch.cat((input, condition), dim=1))


class Discriminator(nn.Module):
    def __init__(self, condition_dim):
        super(Discriminator, self).__init__()
        first_c = 64
        second_c = 128
        third_c = 256
        self.condition_dim = condition_dim
        self.condition_embedding = nn.Embedding(condition_dim, condition_dim)
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, first_c, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(first_c, second_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(second_c),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(second_c, third_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(third_c),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*4) x 4 x 4
        )
        self.main1 = nn.Sequential(
            nn.Conv2d(third_c + condition_dim, 1, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, input, condition):
        input = self.main(input)
        condition = self.condition_embedding(condition)
        condition = condition.unsqueeze(2).unsqueeze(3)
        condition = condition.repeat(1, 1, 4, 4)
        union = torch.cat(tensors=(input, condition), dim=1)
        # union = torch.cat((input.view(input.size(0), -1), condition), dim=1)
        return self.main1(union)





def train(dataloader):
    # Create the generator
    netG = Generator(condition_dim).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)


    # Create the Discriminator
    netD = Discriminator(condition_dim).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.tensor(np.random.normal(0, 1, (8 ** 2, nz)), device="cuda:0", dtype=torch.float)
    fixed_noise = fixed_noise.unsqueeze(2).unsqueeze(3)
    # Get labels ranging from 0 to n_classes for n rows
    fixed_condition = np.array([num for _ in range(8) for num in range(8)])
    fixed_condition = torch.tensor(fixed_condition, device="cuda:0", dtype=torch.long)

    # Establish convention for real and fake labels during training
    real_score = 1.
    fake_score = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x|y)) + log(1 - D(G(z|y)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_image = data[0].to(device)
            real_condition = data[1].to(device)
            b_size = real_image.size(0)
            score = torch.full((b_size,), real_score, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_image, real_condition).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, score)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            d_gen_condition = torch.tensor(np.random.randint(0, condition_dim, b_size), dtype=torch.long,
                                           device=device)
            # Generate fake image batch with G
            fake_image = netG(noise, d_gen_condition)
            score.fill_(fake_score)
            # Classify all fake batch with D
            output = netD(fake_image.detach(), d_gen_condition).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, score)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            score.fill_(real_score)  # fake labels are real for generator cost,减小误差就是接近1，就是最小化V
            # Since we just updated D, perform another forward pass of all-fake batch through D
            g_condition = torch.tensor(np.random.randint(0, condition_dim, b_size), dtype=torch.long,
                                       device=device)
            output = netD(fake_image, g_condition).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, score)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_image = netG(fixed_noise, fixed_condition).detach().cpu()
                img = vutils.make_grid(fake_image, padding=2, normalize=True)
                plt.imshow(np.transpose(img, (1, 2, 0)))
                plt.show()

            iters += 1
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train(loadData())

    # %%capture

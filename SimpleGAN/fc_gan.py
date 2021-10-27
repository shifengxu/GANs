import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
gpu_ids = [2, 3]
device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 1024
num_epochs = 100

disc = Discriminator(image_dim)
genr = Generator(z_dim, image_dim)
fino = torch.randn((batch_size, z_dim))  # fixed noise
if len(gpu_ids) > 1:
    disc = torch.nn.DataParallel(disc, device_ids=gpu_ids)
    genr = torch.nn.DataParallel(genr, device_ids=gpu_ids)
disc = disc.to(device)
genr = genr.to(device)
fino = fino.to(device)
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # (mean, std, inplace=False)
])

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_genr = optim.Adam(genr.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")


# run the training process for a single epoch and a single batch
def run_epoch_batch(epoch, batch_idx, real):
    # real shape is [32, 1, 28, 28] and 32 is the batch size.
    # torch.tensor.view: return a new tensor with the same data but of a different shape.
    # the size -1 is inferred from other dimensions
    real = real.view(-1, 784).to(device)

    # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
    noise = torch.randn(real.shape[0], z_dim).to(device)  # real.shape[0] is batch size
    fake = genr(noise)
    disc_real = disc(real).view(-1)
    disc_fake = disc(fake).view(-1)
    lossD_real = criterion(disc_real, torch.ones_like(disc_real))
    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    lossD = (lossD_real + lossD_fake) / 2
    disc.zero_grad()
    lossD.backward(retain_graph=True)
    opt_disc.step()

    # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
    # where the second option of maximizing doesn't suffer from saturating gradients
    output = disc(fake).view(-1)
    lossG = criterion(output, torch.ones_like(output))
    genr.zero_grad()
    lossG.backward()
    opt_genr.step()

    if batch_idx == 0:
        now = time.time() + 60 * 60 * 8  # seconds. change timezone to UTC+08:00
        dtstr = datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{dtstr} E[{epoch}/{num_epochs}] B[{batch_idx}] LossD:{lossD:.4f} lossG:{lossG:.4f}")
        with torch.no_grad():
            fake = genr(fino).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)
            writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=epoch)
            writer_real.add_image("Mnist Real Images", img_grid_real, global_step=epoch)
        # with
    # if


def main():
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            run_epoch_batch(epoch, batch_idx, real)
    # for


if __name__ == "__main__":
    main()

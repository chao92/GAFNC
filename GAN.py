import argparse
import sys
import os.path as osp
import os
sys.path.insert(1, osp.abspath(osp.join(os.getcwd(), *('..',)*2)))
from dataset_preprocess import CoraDataset
from attack.models import *
import torch
from matplotlib import pyplot as plt
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import seaborn as sns

from pyod.models.copod import COPOD


def build_args():
    parser = argparse.ArgumentParser()
    # data
    # parser.add_argument('--sparsity', type=float, default=0.5, help='sparsity')
    # parser.add_argument('--retrain_epochs', type=int, default=10, help='epochs for retraining a GNN model with new graph')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

    args = parser.parse_args()

    return args

def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)

uniform = torch.distributions.uniform.Uniform(0, 1)


class Ceil(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.ceil(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.clone()
        return output


class Sub(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a - b

    @staticmethod
    def backward(ctx, grad_outputs):
        a, b = ctx.saved_tensors
        return grad_outputs.clone(), torch.zeros_like(b)


class Bernoulli_sample(torch.nn.Module):
    def __init__(self):
        super(Bernoulli_sample, self).__init__()

    def forward(self, input):
        output = Ceil.apply(Sub.apply(input, uniform.sample(input.shape).to(input.device)))
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024, 0.99),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1433),
            nn.BatchNorm1d(1433, 0.99),
            nn.Sigmoid(),
            Bernoulli_sample()
        )


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, z):
        nodes = self.model(z)
        return nodes


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1433, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, node):
        validity = self.model(node)

        return validity


def train_GAN(dataset, generator, optimizer_G, discriminator, optimizer_D):
    losses = []
    adversarial_loss = torch.nn.BCELoss().to(device)

    # batchsize = 20
    batchsize = dataset.data.x.shape[0]
    epoch_n = 500
    counter = 0
    latent_dim = 128
    for epoch in range(epoch_n):
        # for batch in [dataset.data.x]:
        for i in range(0, dataset.data.x.shape[0], batchsize):
            batch = dataset.data.x[i: i + batchsize]
            nodes = Tensor(batch)
            valid = Tensor(nodes.size(0), 1).fill_(1.0).requires_grad_(False)
            fake = Tensor(nodes.size(0), 1).fill_(0.0).requires_grad_(False)

            # Configure input
            real_nodes = nodes.type(Tensor)

            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (nodes.size(0), latent_dim))).requires_grad_(False)

            # Generate a batch of images
            gen_nodes = generator(z)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            if (counter < 25) or (counter % 500 == 0):
                num_critics = 10
            else:
                num_critics = 1

            for _ in range(num_critics):
                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_nodes), valid)
                fake_loss = adversarial_loss(discriminator(gen_nodes.detach()), fake)
                d_loss = real_loss + fake_loss

                d_loss.backward()
                optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_nodes), valid)

            g_loss.backward()
            optimizer_G.step()

            counter += 1

            print("Epoch{}/{}...".format(epoch + 1, epoch_n),
                  "Discriminator Loss:{:.4f}...".format(d_loss),
                  "Generator Loss:{:.4f}...".format(g_loss))

            losses.append((d_loss.item(), g_loss.item()))
    torch.save(generator.state_dict(), 'GAN_model/G.pth')
    torch.save(discriminator.state_dict(), 'GAN_model/D.pth')
    return losses

if __name__ == '__main__':
    args = build_args()
    fix_random_seed(seed=args.seed)
    added_new_nodes = 0
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    mode = "train"
    dataset = CoraDataset('./datasets', 'cora', added_new_nodes=added_new_nodes)
    if cuda:
        dataset.data.x = dataset.data.x.cuda()
    latent_dim = 128

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        # adversarial_loss.cuda()


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr * 10, betas=(args.b1, args.b2 * 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr / 50, betas=(args.b1, args.b2))



    is_train = False
    if is_train:
        losses = train_GAN(dataset, generator, optimizer_G, discriminator, optimizer_D)
    else:
        # Load GAN
        generator.load_state_dict(torch.load('GAN_model/G.pth', map_location=torch.device('cpu')))
        discriminator.load_state_dict(torch.load('GAN_model/D.pth', map_location=torch.device('cpu')))

    _total_detected = 0
    generated_num = 10*20.0
    _random_detected = 0

    clf = COPOD()
    clf.fit(dataset.data.x)
    for i in range(10):
        with torch.no_grad():
            fake_nodes = generator(
                Tensor(np.random.normal(0, 1, (20, latent_dim))).requires_grad_(False))
        res = fake_nodes.detach().cpu()

        random_generate_nodes = np.random.randint(2, size=(20, 1433))
        random_generate_nodes = np.random.choice([0,1], size=(20, 1433), p=[0.5,0.5])
        print("generated nodes is", random_generate_nodes)
        # plt.hist(res.sum(dim=1))
        # sns.distplot(res.sum(dim=1), bins=int(res.sum(dim=1).max()))
        # plt.show()

        print(" start train the COPOD detector")
        y_train_scores = clf.decision_scores_
        y_test_scores = clf.predict(res)
        y_random_scores = clf.predict(random_generate_nodes)
        _total_detected += sum(y_test_scores)
        _random_detected += sum(y_random_scores)

    print("GAN: ratio of detected is", _total_detected/generated_num)
    print("Random: ratio of detected is", _random_detected/generated_num)



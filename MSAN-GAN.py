
from log_print import Log
import torchvision.models as models
import numpy as np
from torchvision import utils
from itertools import chain
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import matplotlib

SAVE_PER_TIMES = 100

def model_load(net, model_path):
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    from scipy import linalg
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



class CNN_T2(torch.nn.Module):

    def __init__(self, model_path):
        super().__init__()

        self.main_module = models.resnet34()
        in_channel = self.main_module.fc.in_features

        self.main_module.fc = nn.Sequential(nn.Linear(in_channel, 224),
                        nn.ReLU(),
                        nn.Linear(224, 2),
                        nn.Softmax(dim=1))
        self.main_module = model_load(self.main_module, model_path)
        self.main_module.eval()
        self.clf = self.main_module.fc

    def forward(self, x):
        x = self.main_module.embedding(x)
        return x


class CNN_DCE(torch.nn.Module):

    def __init__(self, model_path):
        super().__init__()

        self.main_module = models.resnet34()
        self.main_module.conv1 = nn.Conv2d(7, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        in_channel = self.main_module.fc.in_features
        self.main_module.fc = nn.Sequential(nn.Linear(in_channel, 224),
                               nn.ReLU(),
                               nn.Linear(224, 2),
                               nn.Softmax(dim=1))


        self.main_module = model_load(self.main_module, model_path)
        self.main_module.eval()

        self.clf = self.main_module.fc

    def forward(self, x):

        x = self.main_module.embedding(x)
        return x
    


class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128, bias=True)
        )

    def forward(self, x):
        x = self.main_module(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=512, bias=True),
        )

    def forward(self, x):

        x = self.main_module(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.main_module = nn.Sequential(

            nn.Linear(in_features=128, out_features=256, bias=True),

            nn.ReLU(True),

            nn.Linear(in_features=256, out_features=512, bias=True)


        )

    def forward(self, x):
        x = self.main_module(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.main_module = nn.Sequential(



            nn.Linear(in_features=512, out_features=256, bias=True),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=256, out_features=128, bias=True),

            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=128, out_features=1, bias=True))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class AMCCNet(object):
    def __init__(self, args, t2_modelpath, dce_modelpath):
        self.CNN_T2 = CNN_T2(t2_modelpath)
        self.CNN_DCE = CNN_DCE(dce_modelpath)
        self.CNN_T2.eval()
        self.CNN_DCE.eval()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.G = Generator()
        self.D = Discriminator()

        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = args.batch_size


        self.d_optimizer = optim.Adam(
            self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(
            self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))


        self.generator_iters = args.generator_iters
        self.autoencoder_iter = 5
        self.critic_iter = 5
        self.lambda_term = 10





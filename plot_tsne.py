import os
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets

from utils import weights_init
from utils import transform_config

from networks import Encoder, Decoder


def main(FLAGS):
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    encoder.apply(weights_init)

    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder.apply(weights_init)

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        encoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.decoder_save)))

    device = 'cuda:0'

    decoder.to(device)
    encoder.to(device)

    tsne = TSNE(2)

    mnist = DataLoader(datasets.MNIST(root='mnist', download=True, train=False, transform=transform_config))
    s_dict = {}
    with torch.no_grad():
        for i, (image, label) in enumerate(mnist):
            label = int(label)
            print(i, label)
            style_mu_1, style_logvar_1, class_latent_space_1 = encoder(image.to(device))
            s_dict.setdefault(label, []).append(class_latent_space_1)


    s_all = []
    for label in range(10):
        s_all.extend(s_dict[label])

    s_all = torch.cat(s_all)
    s_all = s_all.view(s_all.shape[0], -1).cpu()

    s_2d = tsne.fit_transform(s_all)

    np.savez('s_2d.npz', s_2d=s_2d)

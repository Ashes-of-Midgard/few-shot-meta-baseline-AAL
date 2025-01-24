import argparse
import yaml
import os
import shutil
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # save tsne
    if args.tsne_dir is not None:
        if not os.path.exists(os.path.join('save', args.tsne_dir)):
            os.mkdir(os.path.join('save', args.tsne_dir))
        else:
            shutil.rmtree(os.path.join('save', args.tsne_dir))
            os.mkdir(os.path.join('save', args.tsne_dir))
    with open('imagenet_cat.json', 'r') as f:
        WNID2name = json.load(f)

    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    if not args.sauc:
        n_way = 5
    else:
        n_way = 2
    n_shot, n_query = args.shot, 15
    n_batch = 200
    ep_per_batch = 4
    batch_sampler = CategoriesSampler(
            dataset.label, n_batch, n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=8, pin_memory=True)

    # model
    if config.get('load') is None:
        model = models.make('meta-baseline', encoder=None)
    else:
        model = models.load(torch.load(config['load']))

    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []
    for epoch in range(1, test_epochs + 1):
        for batch_index, (data, label) in tqdm(enumerate(loader), leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)

            with torch.no_grad():
                if not args.sauc:
                    # draw tSNE
                    if args.tsne_dir is not None:
                        feats = model.encoder(x_query.flatten(start_dim=0, end_dim=-4))
                        assert len(feats) % (n_way*n_query) == 0
                        for i in range(len(feats)//(n_way*n_query)):
                            feats_episode = feats[i*n_way*n_query:(i+1)*n_way*n_query]
                            tsne = TSNE(n_components=2).fit_transform(feats_episode.cpu())
                            x_min, x_max = tsne.min(0), tsne.max(0)
                            tsne_norm = (tsne - x_min) / (x_max - x_min)
                            assert len(tsne_norm)==n_query*n_way
                            tsne_split = [tsne_norm[n_query*i:n_query*(i+1)] for i in range(n_way)]
                            colors = ['red', 'green', 'blue', 'brown', 'purple']
                            plt.figure(figsize=(8, 8))
                            label_names = set()
                            for j in range(n_way):
                                label_name = WNID2name[dataset.label2catname[label[i*n_way*n_query+j*n_query].item()]]
                                plt.scatter(tsne_split[j][:, 0], tsne_split[j][:, 1], 50, color=colors[j], label=f'{label_name}')
                                label_names.add(label_name)
                            plt.legend(loc='upper left')
                            if {'mixing bowl', 'malamute', 'scoreboard', 'crate', 'nematode'}.issubset(label_names):
                                plt.savefig(os.path.join('save', args.tsne_dir, f'{label_names}_epoch_{epoch}_batch_{batch_index}.png'))
                            plt.close()
                    logits = model(x_shot, x_query).view(-1, n_way)
                    label = fs.make_nk_label(n_way, n_query,
                            ep_per_batch=ep_per_batch).cuda()
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

                    aves['vl'].add(loss.item(), len(data))
                    aves['va'].add(acc, len(data))
                    va_lst.append(acc)
                else:
                    x_shot = x_shot[:, 0, :, :, :, :].contiguous()
                    shot_shape = x_shot.shape[:-3]
                    img_shape = x_shot.shape[-3:]
                    bs = shot_shape[0]
                    p = model.encoder(x_shot.view(-1, *img_shape)).reshape(
                            *shot_shape, -1).mean(dim=1, keepdim=True)
                    q = model.encoder(x_query.view(-1, *img_shape)).view(
                            bs, -1, p.shape[-1])
                    p = F.normalize(p, dim=-1)
                    q = F.normalize(q, dim=-1)
                    s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
                    for i in range(bs):
                        k = s.shape[1] // 2
                        y_true = [1] * k + [0] * k
                        acc = roc_auc_score(y_true, s[i])
                        aves['va'].add(acc, len(data))
                        va_lst.append(acc)

        print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item(), label[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--tsne_dir', default=None)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)


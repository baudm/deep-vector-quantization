#!/usr/bin/env python3

import torch
from data.str import STRData
from vqvae import VQVAE

BATCH_SIZE=192
class FakeArgs:
    pass
args = FakeArgs()
args.data_dir = '/home/darwin/Projects/thesis/data_lmdb_release'
args.batch_size = BATCH_SIZE
args.num_workers = 8
args.imgH = 32
args.imgW = 100
args.data_filtering_off = False
args.PAD = False
args.sensitive = False
args.character = '0123456789abcdefghijklmnopqrstuvwxyz'
args.rgb = False
args.batch_max_length = 25

data_module = STRData(args)

import sys
args = FakeArgs()
args.vq_flavor = 'vqvae'
args.enc_dec_flavor = 'deepmind'
args.n_hid = 64
args.embedding_dim = 64
args.num_embeddings = 512*2
args.loss_flavor = 'l2'
#vq = VQVAE.load_from_checkpoint('saved/version_22/checkpoints/epoch=3-step=451311.ckpt', args=args, input_channels=1)
vq = VQVAE.load_from_checkpoint(sys.argv[1], args=args, input_channels=1)
vq.cuda()
vq.eval()

codes = []
mse = []

for img, label in data_module.val_dataloader():
    #z = vq.encoder(img)
    #idx = vq.quantizer(z)[-1]
    img = img.cuda()
    with torch.no_grad():
        x_hat, _, idx = vq(img)
    codes.append(idx)
    l = torch.nn.functional.mse_loss(x_hat, img)
    mse.append(l.item())

codes = torch.cat(codes)
with torch.no_grad():
    print(codes.shape)
    perplexity, code_use = vq.compute_metrics(codes)
print('P:', perplexity.item(), 'C:', code_use.item(), 'M:', torch.mean(torch.as_tensor(mse)).item())

"""
The critical quantization layers that we sandwich in the middle of the autoencoder
(between the encoder and decoder) that force the representation through a categorical
variable bottleneck and use various tricks (softening / straight-through estimators)
to backpropagate through the sampling process.
"""

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

import faiss

# -----------------------------------------------------------------------------
class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.0

        self.proj = nn.Conv2d(num_hiddens, embedding_dim, 1)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.embed = nn.Embedding(n_embed, embedding_dim)
        nn.init.normal_(self.embed.weight, 0, 0.1)

        self.i = 1
        self.m_init = 10001
        self.m_reestim = 120000
        self.r_reestim = 5000
        self.reservoir = []
        self.max_cache_size = 1000000 // (200 * 128 // 10) # N - max items
        self.kmeans = faiss.Kmeans(embedding_dim, n_embed, gpu=True)

    def forward(self, z):
        B, C, H, W = z.size()

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
        z_e = self.bn(self.proj(z))
        z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
        flatten = z_e.reshape(-1, self.embedding_dim)

        if self.training:
            # Cache 10% of the samples
            rp = torch.randperm(flatten.shape[0])[:flatten.shape[0] // 10]
            self.reservoir.append(flatten[rp].detach().cpu().numpy().astype(np.float32))
            if len(self.reservoir) >= self.max_cache_size:
                # Remove random item
                self.reservoir.pop(np.random.randint(len(self.reservoir)))

        if self.training and self.i < self.m_init:
            z_q = z_e.detach()
            ind = torch.zeros((B, H, W), dtype=int, device=flatten.device)
        else:
            if self.training and self.i % self.r_reestim == 0 and self.i < self.m_init + self.m_reestim:
                embeddings = np.concatenate(self.reservoir)
                # Reset
                self.reservoir = []
                print('kmeans', embeddings.shape)
                self.kmeans.train(embeddings)
                self.embed.weight.data.copy_(torch.from_numpy(self.kmeans.centroids))

            dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed.weight.t()
                + self.embed.weight.pow(2).sum(1, keepdim=True).t()
            )
            _, ind = (-dist).max(1)
            ind = ind.view(B, H, W)

            # vector quantization cost that trains the embedding vectors
            z_q = self.embed_code(ind) # (B, H, W, C)

        self.i += 1

        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = z_q.permute(0, 3, 1, 2) # stack encodings into channels again: (B, C, H, W)
        return z_q, diff, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        return z_q, diff, ind

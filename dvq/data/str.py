import os

from torch.utils.data import DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl

from .dataset import AlignCollate, hierarchical_dataset


class STRData(pl.LightningDataModule):
    """ returns cifar-10 examples in floats in range [0,1] """

    def __init__(self, args, for_vq_training=True):
        super().__init__()
        self.hparams = args
        self.for_vq_training = for_vq_training

    def _dataloader(self, split, collate_fn):
        root = os.path.join(self.hparams.data_dir, split)
        transform = [
            T.Resize((self.hparams.imgH, self.hparams.imgW), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ]
        if self.for_vq_training and split == 'training':
            transform.extend([
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip()
            ])
        print(transform)
        transform = T.Compose(transform)

        dataset = hierarchical_dataset(root, self.hparams, transform=transform)[0]
        #collate_fn = AlignCollate(imgH=self.hparams.imgH, imgW=self.hparams.imgW, keep_ratio_with_pad=self.hparams.PAD)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=split == 'training',
            collate_fn=collate_fn
        )
        return dataloader

    def train_dataloader(self, collate_fn=None):
        return self._dataloader('training', collate_fn)

    def val_dataloader(self, collate_fn=None):
        return self._dataloader('validation', collate_fn)

    def test_dataloader(self, collate_fn=None):
        return self._dataloader('evaluation', collate_fn)

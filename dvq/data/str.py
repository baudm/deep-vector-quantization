import os

from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torchvision import transforms as T

import pytorch_lightning as pl

from .dataset import AlignCollate, hierarchical_dataset
from .sampler import BatchBalancedSampler


class STRData(pl.LightningDataModule):
    """ returns cifar-10 examples in floats in range [0,1] """

    @staticmethod
    def add_data_specific_args(parser):
        group = parser.add_argument_group('STRData')
        # dataloader related
        group.add_argument("--data_dir", type=str, default='/apcv/users/akarpathy/cifar10')
        group.add_argument("--batch_size", type=int, default=128)
        group.add_argument("--num_workers", type=int, default=8)
        # dataset related
        group.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        group.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        group.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        group.add_argument('--rgb', action='store_true', help='use rgb input')
        group.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                           help='character label')
        group.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        group.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        group.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
        return parser

    def __init__(self, args, collate_fn=None, for_vq_training=True):
        super().__init__()
        self.hparams = args
        self.collate_fn = collate_fn
        self.for_vq_training = for_vq_training

    def _dataloader(self, split):
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

        if split == 'training':
            samplers = []
            datasets = []
            for d in ['MJ', 'ST']:
                dataset = hierarchical_dataset(root, self.hparams, select_data=d, transform=transform)[0]
                datasets.append(dataset)
                samplers.append(DistributedSampler(dataset))
            r = [1., 1.]
            dataset = ConcatDataset(datasets)
            sampler = BatchBalancedSampler(samplers, r, self.hparams.batch_size, False)
        else:
            if split == 'evaluation':
                subset = ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
            else:
                subset = '/'
            dataset = hierarchical_dataset(root, self.hparams, select_data=subset, transform=transform)[0]
            sampler = None

        #collate_fn = AlignCollate(imgH=self.hparams.imgH, imgW=self.hparams.imgW, keep_ratio_with_pad=self.hparams.PAD)
        dataloader = DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=self.collate_fn
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader('training')

    def val_dataloader(self):
        return self._dataloader('validation')

    def test_dataloader(self):
        return self._dataloader('evaluation')
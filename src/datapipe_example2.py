import os
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
from typing import Tuple
import numpy as np
import torch
import torchdata

import utils
from config import Config
import dataset_example


def prepare_data():
    cfg = Config()
    cfg.transform = None
    os.makedirs(cfg.prepared_data_path, exist_ok=True)
    celeba_dataset = dataset_example.CelebADataset(f'{cfg.data_path}/CelebA', cfg.transform)
    digiface_dataset = dataset_example.DigiFace1M(f'{cfg.data_path}/DigiFace1M', cfg.transform, cfg.n_celeba_classes)
    dataset = torch.utils.data.ConcatDataset([celeba_dataset, digiface_dataset])

    shard_size = 10000
    next_shard = 0
    data = []
    shuffled_idxs = np.arange(len(dataset))
    np.random.shuffle(shuffled_idxs)
    for idx in tqdm(shuffled_idxs):
        data.append(dataset[idx])
        if len(data) == shard_size:
            with open(f'{cfg.prepared_data_path}/{next_shard}_shard.pickle', 'wb') as _file:
                pickle.dump(data, _file)
            next_shard += 1
            data = []
    with open(f'{cfg.prepared_data_path}/{next_shard}_shard.pickle', 'wb') as _file:
        pickle.dump(data, _file)


@torchdata.datapipes.functional_datapipe("load_pickle_data")
class PickleDataLoader(torchdata.datapipes.iter.IterDataPipe):
    def __init__(self, source_datapipe, **kwargs) -> None:
        self.source_datapipe = source_datapipe
        self.transform = kwargs['transform']

    def __iter__(self) -> Tuple[torch.Tensor, int]:
        for file_name in self.source_datapipe:
            with open(file_name, 'rb') as _file:
                pickle_data = pickle.load(_file)
                for image, label in pickle_data:
                    image = self.transform(image)
                    yield image, label


def build_datapipes(cfg: Config) -> torchdata.datapipes.iter.IterDataPipe:
    datapipe = torchdata.datapipes.iter.FileLister(cfg.prepared_data_path, masks='*.pickle')
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.load_pickle_data(transform=cfg.transform)
    return datapipe


def main():
    parser = ArgumentParser()
    parser.add_argument('--prepare_data', action='store_true', help='Create sharding files')
    args = parser.parse_args()
    if args.prepare_data:
        prepare_data()

    cfg = Config()
    datapipe = build_datapipes(cfg)
    loader = torch.utils.data.DataLoader(
        dataset=datapipe,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.n_workers)

    utils.train(loader, cfg)


if __name__ == '__main__':
    main()

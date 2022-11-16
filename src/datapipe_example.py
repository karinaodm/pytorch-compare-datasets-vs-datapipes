import os
from typing import Dict, Tuple
from PIL import Image
import torch
import torchdata

import utils
from config import Config


N_CELEBA_CLASSES = 10177


@torchdata.datapipes.functional_datapipe("load_image")
class ImageLoader(torchdata.datapipes.iter.IterDataPipe):
    def __init__(self, source_datapipe, **kwargs) -> None:
        self.source_datapipe = source_datapipe
        self.transform = kwargs['transform']

    def __iter__(self) -> Tuple[torch.Tensor, int]:
        for file_name, label, data_name in self.source_datapipe:
            image = Image.open(file_name)
            if data_name == 'DigiFace1M':
                image = image.convert('RGB')
            elif data_name == 'CelebA':
                left, right, top, bottom = 25, 153, 45, 173
                image = image.crop((left, top, right, bottom))
            if self.transform is not None:
                image = self.transform(image)
            yield image, label


def collate_ann(file_path):
    label = int(os.path.basename(os.path.dirname(file_path))) + N_CELEBA_CLASSES
    data_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    return file_path, label, data_name


def load_celeba_labels(labels_path: str) -> Dict[str, int]:
    labels = []
    data_path = os.path.split(labels_path)[0]
    with open(labels_path, 'r', encoding='utf-8') as labels_file:
        lines = labels_file.readlines()
        for line in lines:
            file_name, class_id = line.split(' ')
            class_id = int(class_id[:-1])
            labels.append((f'{data_path}/img_align_celeba/{file_name}', class_id, 'CelebA'))
    return labels


def build_datapipes(cfg: Config) -> torchdata.datapipes.iter.IterDataPipe:
    celeba_dp = torchdata.datapipes.iter.IterableWrapper(
        load_celeba_labels(
            labels_path=f'{cfg.data_path}/CelebA/identity_CelebA.txt'))

    digiface_dp = torchdata.datapipes.iter.FileLister(f'{cfg.data_path}/DigiFace1M', masks='*.png', recursive=True)
    digiface_dp = digiface_dp.map(collate_ann)

    datapipe = celeba_dp.concat(digiface_dp)
    datapipe = datapipe.shuffle(buffer_size=100000)
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.load_image(transform=cfg.transform)
    return datapipe


def main():
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

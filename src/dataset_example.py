import os
from typing import Tuple
from PIL import Image
import torch

import utils
from config import Config


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, transform) -> None:
        self.data_path = data_path
        self.transform = transform
        self.image_names, self.labels = self.load_labels(f'{data_path}/identity_CelebA.txt')

    def __len__(self) -> int:
        return len(self.image_names)

    def  __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = f'{self.data_path}/img_align_celeba/{self.image_names[idx]}'
        image = Image.open(image_path)
        left, right, top, bottom = 25, 153, 45, 173
        image = image.crop((left, top, right, bottom))
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    @staticmethod
    def load_labels(labels_path: str) -> Tuple[list, list]:
        image_names, labels = [], []
        with open(labels_path, 'r', encoding='utf-8') as labels_file:
            lines = labels_file.readlines()
            for line in lines:
                file_name, class_id = line.split(' ')
                image_names.append(file_name)
                labels.append(int(class_id[:-1]))
        return image_names, labels


class DigiFace1M(torch.utils.data.Dataset):
    def __init__(self, data_path: str, transform, add_to_class: int = 0) -> None:
        self.data_path = data_path
        self.transform = transform
        self.image_paths, self.labels = self.load_labels(data_path, add_to_class)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    @staticmethod
    def load_labels(data_path: str, add_to_class: int) -> Tuple[list, list]:
        image_paths, labels = [], []
        for root, _, files in os.walk(data_path):
            for file_name in files:
                if file_name.endswith('.png'):
                    image_paths.append(f'{root}/{file_name}')
                    labels.append(int(os.path.basename(root)) + add_to_class)
        return image_paths, labels


def main():
    cfg = Config()
    celeba_dataset = CelebADataset(f'{cfg.data_path}/CelebA', cfg.transform)
    digiface_dataset = DigiFace1M(f'{cfg.data_path}/DigiFace1M', cfg.transform, cfg.n_celeba_classes)
    dataset = torch.utils.data.ConcatDataset([celeba_dataset, digiface_dataset])

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.n_workers)

    utils.train(loader, cfg)


if __name__ == '__main__':
    main()

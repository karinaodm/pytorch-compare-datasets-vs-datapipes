from dataclasses import dataclass
from torchvision import transforms


@dataclass
class Config:
    data_path = 'data'
    prepared_data_path = 'prepared_data'
    epochs: int = 1
    batch_size: int = 600
    n_workers: int = 5
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.RandomCrop((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    n_celeba_classes: int = 10177
    n_digiface1m_classes: int = 10000
    gpu: int = 0
    use_amp: bool = True

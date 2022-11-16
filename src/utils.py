import time
import torch

from resnet import resnet50
from config import Config


def train(data_loader: torch.utils.data.DataLoader, cfg: Config):
    # create model
    model = resnet50(num_classes=cfg.n_celeba_classes + cfg.n_digiface1m_classes, pretrained=True)
    torch.cuda.set_device(cfg.gpu)
    model = model.cuda(cfg.gpu)
    model.train()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(cfg.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9,
                                weight_decay=1e-4)

    start_time = time.time()
    for _ in range(cfg.epochs):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        for batch_idx, (images, target) in enumerate(data_loader):
            images = images.cuda(cfg.gpu, non_blocking=True)
            target = target.cuda(cfg.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                output = model(images)
                loss = criterion(output, target)

            # compute gradient
            scaler.scale(loss).backward()

            # do SGD step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            print(batch_idx, loss.item())
    print(f'{time.time() - start_time} sec')

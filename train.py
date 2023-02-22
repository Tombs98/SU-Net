import argparse
import os
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import UNet
from data import MyDatset
import wandb
from model.res_unet import Resnet34_Unet
# from tensorboardX import SummaryWriter
import numpy as np
# from model.res_net import ResNet50
from model.alexnet import AlexNet
from model.vgg import VGG_19
# writer = SummaryWriter('runs/fashion_mnist_experiment_1')
from model.bottleneckTrans import ResNet50
from vit_pytorch import ViT

dir_checkpoint = Path('./vgg/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 8,
              learning_rate: float = 0.005,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    train_loader, val_loader = MyDatset.split_dataset()

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.L1Loss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        sum3 = 0
        loss3 = 0
        length = len(train_loader)
        net.train()
        epoch_loss = 0
        with tqdm(desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch[0]
                true_label = batch[1]
                # assert images.shape[1] == net.n_channels, \
                #   f'Network has been defined with {net.n_channels} input channels, ' \
                #  f'but loaded images have {images.shape[1]} channels. Please check that ' \
                # 'the images are loaded correctly.'
                images = images.to(device=device, dtype=torch.float32)
                true_label = true_label.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # print("sssss")
                    # print(true_label)
                    # print(true_label.shape)
                    # print("ssssssssssss")
                    # print(masks_pred)
                    # print(masks_pred.shape)
                    # break
                    sum3 = sum3 + abs(masks_pred - true_label)
                    loss = criterion(masks_pred, true_label)
                    loss3 = loss + loss3

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                # scheduler.step(loss)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # writer.add_scalar("loss", epoch_loss, epoch * len(train_loader))
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        mean = sum3 / length
        print("mean:", mean)
        mean = mean.cpu().detach().numpy()
        with open("error_mean.txt", "a") as f:
            f.write(str(mean))
        mean_loss = loss3 / length
        print("loss:", mean_loss)
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.4, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # net = UNet(n_channels=3, n_classes=1, bilinear=True)
    # net = ResNet50()
    net = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    # net = Resnet34_Unet(in_channel=3, out_channel=1)
    # logging.info(f'Network:\n'
    #            f'\t{net.n_channels} input channels\n'
    #          f'\t{net.n_classes} output channels (classes)\n'
    #        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    # print(torch.cuda.device_count())
    # net = nn.parallel.DistributedDataParallel(net)
    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)

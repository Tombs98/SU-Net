import os
import torch
import torch.nn as nn
from data import MyDatset
from model.res_unet import Resnet34_Unet

def test(net, device, epochs):
    train_loader, val_loader = MyDatset.split_dataset()
    for epoch in range(epochs):
        sum3 = 0
        loss3 = 0
        length = len(val_loader)
        for batch in val_loader:
            images = batch[0]
            true_label = batch[1]
                # assert images.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'
            images = images.to(device=device, dtype=torch.float32)
            true_label = true_label.to(device=device, dtype=torch.float32)
            masks_pred = net(images)
            sum3 = sum3 + abs(masks_pred - true_label)
        mean = sum3 / length
        print("mean:", mean)
        print("mean:", torch.mean(mean))

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Resnet34_Unet(3,1, False)
    net.load_state_dict(torch.load("./module/res-u-net.pth"))
    net.to(device)
    print(net)
    with torch.no_grad():
    	test(net, device, 5)
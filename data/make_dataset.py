from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from .obtain import Obtain_data
from PIL import Image

img_path = Obtain_data.t_list()
deg_label = Obtain_data.deg_list()
deg_label = np.asanyarray(deg_label)
deg_label = deg_label.astype('float')


class MyDatset(Dataset):
    def __init__(self, deg_label, img_path, transform=None, target_transform=None):
        imgs = []
        for i in range(len(deg_label)):
            imgs.append((img_path[i], deg_label[i]))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        t_path, deg_label = self.imgs[item]
        img = Image.open(t_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, deg_label

    def __len__(self):
        return len(self.imgs)

    def split_dataset():
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(256),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        dataset = MyDatset(deg_label=deg_label, img_path=img_path, transform=data_transform)
        print('total_size', len(dataset))
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])

        print('****************************')
        print('test_size', len(test_data))
        print('train_size', len(train_data))

        print('*******************')

        train_loader = DataLoader(dataset = train_data, batch_size=2, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset = test_data, batch_size=1)
        return train_loader, test_loader


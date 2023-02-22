from .vgg_parts import VGG

class VGG_():
    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def vgg11(**kwargs):
        model = VGG(make_layers(cfg['A']), **kwargs)
        return model

    def vgg11_bn(**kwargs):
        model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
        return model

    def vgg13(**kwargs):
        model = VGG(make_layers(cfg['B']), **kwargs)
        return model

    def vgg13_bn(**kwargs):
        model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
        return model

    def vgg16(**kwargs):
        model = VGG(make_layers(cfg['D']), **kwargs)
        return model

    def vgg16_bn(**kwargs):
        model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
        return model

    def vgg19(**kwargs):
        model = VGG(make_layers(cfg['E']), **kwargs)
        return model

    def vgg19_bn(**kwargs):
        model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
        return model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision.models.densenet import model_urls, _DenseLayer, _DenseBlock, _Transition

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # default: kernel_size=7 | changed to 1 for smaller images | >=61
        out = F.avg_pool2d(out, kernel_size=1, stride=1).view(features.size(0), -1)
        #out = self.classifier(out)
        return out


def extract_feature_state_dict(pretrained_state_dict, model):
    model_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
    return pretrained_state_dict



def densenet121(pretrained=False, requires_grad=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['densenet121'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def densenet169(pretrained=False, requires_grad=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['densenet169'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def densenet201(pretrained=False, requires_grad=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['densenet201'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def densenet161(pretrained=False, requires_grad=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['densenet161'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model

def test_densenet(size, dense):
    from torch.autograd import Variable
    dense = int(dense)
    if dense==121:
        net = densenet121(pretrained=True)
    elif dense==161:
        net = densenet161(pretrained=True)
    elif dense==169:
        net = densenet169(pretrained=True)
    elif dense==201:
        net = densenet201(pretrained=True)
    x = torch.randn(1,3,size,size)
    y = net(Variable(x))
    print(y.shape)

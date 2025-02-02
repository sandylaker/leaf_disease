import torch
import torch.nn as nn
from .efficientnet_utils import MBConv, Flatten, Swish, NormedLinear
from collections import OrderedDict
import math


def init_weights(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


class EfficientNet(nn.Module):
    # maps scale to coefficients of (width, depth, dropout)
    param_dict = {
        0: (1.0, 1.0, 0.2),
        1: (1.0, 1.1, 0.2),
        2: (1.1, 1.2, 0.3),
        3: (1.2, 1.4, 0.3),
        4: (1.4, 1.8, 0.4),
        5: (1.6, 2.2, 0.4),
        6: (1.8, 2.6, 0.5),
        7: (2.0, 3.1, 0.5),
    }

    def __init__(self,
                 in_channels: int = 3,
                 n_classes: int = 13,
                 scale: int = 0,
                 se_rate: float = 0.25,
                 drop_connect_rate: float = 0.2,
                 normed_fc: dict = None,
                 uniform_fc_bias: bool = False,
                 pretrained=None,
                 frozen_stages: int = -1):
        super(EfficientNet, self).__init__()
        assert scale in range(0, 8)
        self.frozen_stages = frozen_stages
        self.in_channels = in_channels
        width_coefficient, depth_coefficient, dropout_rate = self.param_dict[scale]
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8
        self.n_classes = n_classes
        self.normed_fc = normed_fc
        self.uniform_fc_bias = uniform_fc_bias

        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]
        self.list_channels = list_channels

        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]
        self.list_num_repeats = list_num_repeats

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        # Define stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.list_channels[0], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.list_channels[0], momentum=0.01, eps=1e-3),
            Swish()
        )

        # Define blocks
        blocks = []
        counter = 0
        num_blocks = sum(self.list_num_repeats)
        for idx in range(7):
            num_channels = self.list_channels[idx]
            next_num_channels = self.list_channels[idx + 1]
            num_repeats = self.list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            # drop rate increases as depth increases
            drop_rate = drop_connect_rate * counter / num_blocks

            name = 'MBConv{}_{}'.format(expand_rate, counter)
            blocks.append((
                name,
                MBConv(num_channels, next_num_channels, kernel_size=kernel_size,
                       stride=stride, expand_rate=expand_rate, se_rate=se_rate,
                       drop_connect_rate=drop_rate)
            ))
            counter += 1
            for i in range(1, num_repeats):
                name = 'MBConv{}_{}'.format(expand_rate, counter)
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append((
                    name,
                    MBConv(next_num_channels, next_num_channels, kernel_size=kernel_size,
                           stride=1, expand_rate=expand_rate, se_rate=se_rate,
                           drop_connect_rate=drop_rate)
                ))
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

        # Define head
        self.head = nn.Sequential(
            nn.Conv2d(self.list_channels[-2], self.list_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(self.list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.list_channels[-1], self.n_classes))
        if self.normed_fc is not None:
            self.head[-1] = NormedLinear(self.list_channels[-1], self.n_classes, **self.normed_fc)

        self.init_weights(pretrained=pretrained)
        self.freeze_stages()

    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        """To ensure the new number of channels can be divided by divisor, for example 8."""
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        # To ensure the new number of channels are greater or equal to divisor
        new_num_channels = max(self.divisor, new_num_channels)
        # To avoid number of channels shrink too much
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            incompatible_keys = load_checkpoint(self, pretrained)
        elif pretrained is None or not pretrained:
            self.apply(init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        # NormedLinear has no bias
        if self.uniform_fc_bias and not self.normed_fc:
            self._init_fc_bias()

    def _init_fc_bias(self):
        pi = 1 / self.n_classes
        bias = - math.log((1 - pi) / pi)
        nn.init.constant_(self.head[-1].bias, bias)

    def freeze_stages(self):
        """Freeze the parameters at stages. Let f denote the value of self.frozen_edges.
            1. f < 0: no stage will be frozen.
            2. f = 0: stem will be frozen.
            3. 1 <= f <= 7: stem and 0 ~ (f - 1)-th MBConv (0-indexed) will be frozen.
            4. f = 8: stem, all MBConv and head.0~5 will be frozen.
            5. f >= 9: all the layers will be frozen.
        """
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
            idx = 0
            for i in range(0, min(self.frozen_stages, len(self.list_num_repeats))):
                for _ in range(self.list_num_repeats[i]):
                    self.blocks[idx].eval()
                    for param in self.blocks[idx].parameters():
                        param.requires_grad = False
                    idx += 1

            num_frozen_layers_head = self.frozen_stages - len(self.list_num_repeats)
            if num_frozen_layers_head > 0:
                # the head contains 2 stages: before fc + after fc
                num_frozen_layers_head += 5
                for i in range(min(num_frozen_layers_head, 7)):
                    for param in self.head[i].parameters():
                        param.requires_grad = False

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y


def load_checkpoint(model: EfficientNet, pretrained: str):
    model_state = torch.load(pretrained)

    mapping = {k: v for k, v in zip(model_state.keys(), model.state_dict().keys())}

    try:
        # first try to load the checkpoint assuming the heads have the same shapes
        mapped_model_state = OrderedDict([
            (mapping[k], v) for k, v in model_state.items()
        ])
        imcompatible_keys = model.load_state_dict(mapped_model_state, strict=False)
    except:
        # If incompatible head, just load the parts except for the last fc layer
        mapped_model_state = OrderedDict([
            (mapping[k], v) for k, v in model_state.items() if not mapping[k].startswith('head.6')
        ])
        imcompatible_keys = model.load_state_dict(mapped_model_state, strict=False)
        
    print('Loaded ImageNet weights for EfficientNet')
    print('Incompatible keys:', imcompatible_keys)
    return imcompatible_keys

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Act(nn.Module):
    def __init__(self, out_planes=None, act_type="relu", inplace=True):
        super(Act, self).__init__()

        self.act = None
        if act_type == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == "prelu":
            self.act = nn.PReLU(out_planes)
        elif act_type == "hardswish":
            self.act = nn.Hardswish(inplace=True)
        elif act_type == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act_type == "gelu":
            self.act = nn.GELU()

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        return x

class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_type="relu", drop_rate=0.2):
        super().__init__()
        hidden_features = min(2048, int(mlp_ratio * dim))
        self.fc1 = nn.Linear(dim, hidden_features, bias=False)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.act = Act(hidden_features, act_type)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class SE(nn.Module):
    def __init__(self, dim, ratio=8):
        super().__init__()
        hidden_dim = max(8, dim // ratio)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, padding=None, act_type="relu"):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2 if padding is None else padding, bias=False)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = Act(out_planes, act_type)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class CNNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=1.0, stride=1, act_type="relu", drop_path=0.0):
        super(CNNBlock, self).__init__()
        mid_planes = int(out_planes * ratio)
        self.main = nn.Sequential(
            ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, act_type=act_type),
            ConvX(mid_planes, mid_planes, groups=mid_planes//4, kernel_size=3, stride=stride, act_type=act_type),
            SE(mid_planes),
            ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None),
        )

        self.skip = nn.Identity()
        if stride == 2:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes//4, kernel_size=3, stride=2, act_type=None),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.main(x)) + self.skip(x)
        return x


class RegNetY(nn.Module):
    # pylint: disable=unused-variable
    def __init__(self, dims, layers, ratio=1.0, act_type="relu", drop_path_rate=0., num_classes=1000):
        super(RegNetY, self).__init__()
        self.ratio = ratio
        self.act_type = act_type
        self.drop_path_rate = drop_path_rate

        if isinstance(dims, int):
            dims = [dims//2, dims, dims*2, dims*4, dims*8]
        else:
            dims = [dims[0]//2] + dims

        self.first_conv = ConvX(3, dims[0], 1, 3, 2, act_type=act_type)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        self.layer1 = self._make_layers(dims[0], dims[1], layers[0], stride=2, drop_path=dpr[:layers[0]])
        self.layer2 = self._make_layers(dims[1], dims[2], layers[1], stride=2, drop_path=dpr[layers[0]:sum(layers[:2])])
        self.layer3 = self._make_layers(dims[2], dims[3], layers[2], stride=2, drop_path=dpr[sum(layers[:2]):sum(layers[:3])])
        self.layer4 = self._make_layers(dims[3], dims[4], layers[3], stride=2, drop_path=dpr[sum(layers[:3]):sum(layers[:4])])

        head_dim = max(1024, dims[4])
        self.head = ConvX(dims[4], head_dim, 1, 1, 1, act_type=act_type)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = MlpHead(head_dim, num_classes, act_type=act_type)

        self.init_params(self)

    def _make_layers(self, inputs, outputs, num_block, stride, drop_path):
        layers = [CNNBlock(inputs, outputs, self.ratio, stride, self.act_type, drop_path[0])]

        for i in range(1, num_block):
            layers.append(CNNBlock(outputs, outputs, self.ratio, 1, self.act_type, drop_path[i]))

        return nn.Sequential(*layers)

    def init_params(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = self.head(x)
        out = self.gap(out).flatten(1)
        out = self.classifier(out)
        return out

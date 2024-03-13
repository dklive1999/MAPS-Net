import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import numpy as np

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class EdgeAttentionModule(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
        super(EdgeAttentionModule, self).__init__()
        self.mid_channels = mid_channels
        self.f_self = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)

    def forward(self, x, y):
        batch_size = x.size(0)
        fself = self.f_self(x).view(batch_size, self.mid_channels, -1)
        fself = fself.permute(0, 2, 1)
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1)
        fx = fx.permute(0, 2, 1)
        fy = self.f_y(y).view(batch_size, self.mid_channels, -1)
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1)
        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        out = self.f_up(fout)
        return  out



model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, n_input=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        rates = [1, 2, 4]
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)  # stride 2 => stride 1
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(pretrained=False, layers=[3,4,6,3], backbone='resnet50', n_input=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, layers, n_input=n_input, **kwargs)

    pretrain_dict = model_zoo.load_url(model_urls[backbone])
    try:
        model.load_state_dict(pretrain_dict,strict=False)
    except:
        print("loss conv1")
        model_dict = {}
        for k, v in pretrain_dict.items():
            if k in pretrain_dict and 'conv1' not in k:
                model_dict[k] = v
        model.load_state_dict(model_dict, strict=False)
    print("load pretrain success")
    return model


class ResNet50(nn.Module):
    def __init__(self, pretrained=True,n_input=3):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet(n_input=n_input, pretrained=pretrained, layers=[3, 4, 6, 3], backbone='resnet50')
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def base_forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out

class EEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EEB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x+res


class BasicBlock_ups(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_ups, self).__init__()

        self.ups = nn.Sequential()
        if stride == 2:
            self.ups = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.ups(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes,stride=1, kernel_size=3):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
class MAPSNet(ResNet50):
    def __init__(self, nclass,  BatchNorm=nn.BatchNorm2d,aux=False, sobel=False, constrain=False, n_input=3, **kwargs):
        super(MAPSNet, self).__init__(pretrained=True, n_input=n_input)
        self.num_class = nclass
        self.aux = aux

        self.__setattr__('exclusive', ['head'])
        self.BatchNorm = BatchNorm
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.sobel = sobel
        self.constrain = constrain

        self.erb_db_1 = EEB(256, 256)
        self.erb_db_2 = EEB(512, 256)
        self.erb_db_3 = EEB(1024, 256)
        self.erb_db_4 = EEB(2048, 256)
        if self.sobel:
            print("----------use sobel-------------")
            self.sobel_x1, self.sobel_y1 = get_sobel(256, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(512, 1)
            self.sobel_x3, self.sobel_y3 = get_sobel(1024, 1)
            self.sobel_x4, self.sobel_y4 = get_sobel(2048, 1)

#         if self.constrain:
#             print("----------use constrain-------------")
#             self.noise_extractor = ResNet50(n_input=3, pretrained=True)
#             self.constrain_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
        self.head = _DAHead(2048+2048, self.num_class, aux, **kwargs)
        self.edge_attention = EdgeAttentionModule(2048, 256, 256, BatchNorm)
        self.edge_down = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, stride=4,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.myin_planes = 3
        self.mylayer1 = self._make_layer(BasicBlock2, 32, 1, stride=2, kernel_size=3)  # 256，
        self.mylayer2 = self._make_layer(BasicBlock2, 64, 1, stride=2, kernel_size=3)  # 128,
        self.mylayer3 = self._make_layer(BasicBlock2, 128, 2, stride=2, kernel_size=3)  # 64,
        self.mylayer4 = self._make_layer(BasicBlock2, 256, 2, stride=2, kernel_size=3)  # 32,

        self.myUpsamplelayer1 = self._make_layer_ups(BasicBlock_ups, 128, 2,
                                                   stride=2)  # 64,BasicBlock_ups(in_planes=512, planes=256, stride=2)
        self.myUpsamplelayer2 = self._make_layer_ups(BasicBlock_ups, 64, 1,
                                                   stride=2)  # 64,BasicBlock_ups(in_planes=256, planes=64, stride=1)

        self.myconv_layer = nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1)
        self.mymodel = resnet(n_input=n_input,layers=[3, 4, 6, 3], backbone='resnet50')
        mylayers_cfg = [4, 5, 6, 7]
        self.myblocks = []
        for i, num_this_layer in enumerate(mylayers_cfg):
            self.myblocks.append(list(self.mymodel.children())[num_this_layer])

    def _make_layer_2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)
    
    def _make_layer(self, block, planes, num_blocks, stride, kernel_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.myin_planes, planes, stride, kernel_size))
            self.myin_planes = planes * block.expansion
            return nn.Sequential(*layers)

    def _make_layer_ups(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.myin_planes, planes, stride))
            self.myin_planes = planes * block.expansion
        return nn.Sequential(*layers)

    #################################################################
    def forward(self, x):
        out1=  self.mylayer1(x)
        out2 = self.mylayer2(out1)
        out3 = self.mylayer3(out2)
        out4 = self.mylayer4(out3)
        out33 = self.myUpsamplelayer1(out4)
        out22 = self.myUpsamplelayer2(out33)
        myout = torch.abs(out22 -out2)
        my_feature_map = []
        myret2 = torch.mean(myout)
        for i, block in enumerate(self.myblocks):
            if i == 0:
                myx = block(myout)
            else:
                myx = block(myx)
            my_feature_map.append(myx)

        mya1, mya2, mya3, mya4 = my_feature_map

        size = x.size()[2:]
        input_ = x.clone()

        feature_map, _ = self.base_forward(input_)
        c1, c2, c3, c4 = feature_map
        if self.sobel:
            res1 = self.erb_db_1(run_sobel(self.sobel_x1, self.sobel_y1, c1))
            res2 = self.erb_db_2(run_sobel(self.sobel_x2, self.sobel_y2, c2))
            res2 = F.interpolate(res2, scale_factor=2, mode='bilinear', align_corners=True)
            res3 = self.erb_db_3(run_sobel(self.sobel_x3, self.sobel_y3, c3))
            res3 = F.interpolate(res3, scale_factor=4, mode='bilinear', align_corners=True)
            res4 = self.erb_db_4(run_sobel(self.sobel_x4, self.sobel_y4, c4))
            res4 = F.interpolate(res4, scale_factor=4, mode='bilinear', align_corners=True)

        else:
            res1 = self.erb_db_1(c1)
            res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(c2)))
            res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(c3)))
            res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(c4)), relu=False)
        
        E = torch.cat((res1, res2, res3, res4), dim=1)
        E = self.edge_down(E)
        out_feature = self.edge_attention(c4, E)
        
        # Final output
        edge= self.myconv_layer(out_feature)
        edge= F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=True)

        h4 = torch.cat([mya4,c4], dim=1)

        outputs = []

        x = self.head(h4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        outputs.append(x0)

        if self.aux:
            x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            outputs.append(x1)
            outputs.append(x2)

        return edge, x0,myret2


def shift(x, gamma=1/12):
    # x is a tensor with a shape of
    # [Batch, Channel, Height, Width]
    B, C, H, W = x.shape
    g = int(gamma * C)
    out = torch.zeros_like(x)
    # spatially shift
    out[:, 0*g:1*g, :, :-1] = x[:, 0*g:1*g, :, 1:] # shift left
    out[:, 1*g:2*g, :, 1:] = x[:, 1*g:2*g, :, :-1] # shift right
    out[:, 2*g:3*g, :-1, :] = x[:, 2*g:3*g, 1:, :] # shift up
    out[:, 3*g:4*g, 1:, :] = x[:, 3*g:4*g, :-1, :] # shift down
    # remaining channels
    out[:, 4*g:, :, :] = x[:, 4*g:, :, :] # no shift
    return out


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = shift(x)
#         batch_size, _, height, width = x.size()
#         feat_a = x.view(batch_size, -1, height * width)
#         feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
#         attention = torch.bmm(feat_a, feat_a_transpose)
#         attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
#         attention = self.softmax(attention_new)

#         feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
#         out = self.beta * feat_e + x
        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


def get_maps(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, n_input=3, constrain=False, **kwargs):
    model = MAPSNet(nclass, backbone=backbone,
                    pretrained_base=pretrained_base,
                    sobel=sobel,
                    n_input=n_input,
                    constrain=constrain,
                    **kwargs)
    return model


if __name__ == '__main__':
    img = torch.randn(2, 3, 512, 512)
    model = get_maps(sobel=True, n_input=3,pretrained_base=True, constrain=True)
    edge, outputs,ret2 = model(img)
    print(outputs.shape)
    print(edge.shape)
import torch.nn as nn
import torch
import torch.nn.functional as F

class BidimensionalSpatialAttention(nn.Module):
    def __init__(self, in_ch, out_ch, hight):
        super(BidimensionalSpatialAttention, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.query_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.key_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.value_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.factor = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv1d(hight, hight, kernel_size=3,stride=1,padding=1, bias=False)  # 一维卷积

        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_ch // 16, in_ch, 1, bias=False),
        )
        self.conv_1x1_2 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_ch)
        self.conv_1x1= nn.Conv2d(in_ch * 3, out_ch, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        q = self.channel_attention(F.adaptive_avg_pool2d(self.conv_3x3(x),[height,1]))
        k = self.channel_attention(F.adaptive_avg_pool2d(x,[1,width]))


        v1 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//2, width//2]))))
        v1 = F.interpolate(v1, size=x.size()[2:], mode='bilinear', align_corners=True)


        v2 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//6, width//6]))))
        v2 = F.interpolate(v2, size=x.size()[2:], mode='bilinear', align_corners=True)

        v3 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//8, width//8]))))
        v3 = F.interpolate(v3, size=x.size()[2:], mode='bilinear', align_corners=True)

        value = torch.cat([v1,v2,v3],dim=1)
        value = self.conv_1x1(value)
        attention_scores = torch.matmul(q, k)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.out_channels//2, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attended_value = torch.matmul(attention_weights, value)
        output = x + self.factor * attended_value

        return output




class BiSEM(nn.Module):
    def __init__(self, in_channels, out_channels,height):
        super(BiSEM, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)
        self.BiSA = BidimensionalSpatialAttention(in_channels,out_channels,height)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, bias=False)  # 一维卷积
    def forward(self, x):
        x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        x2 = self.BiSA(x)
        x = torch.cat((x1,x2), dim=1)
        x = self.conv2(x)
        return x
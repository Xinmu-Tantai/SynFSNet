import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import math
import numpy as np
import cv2

def blc2bchw(x, h, w):
    b, l, c = x.shape
    assert l == h * w, "in blc to bchw, h*w != l."
    return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def bchw2blc(x, h, w):
    b, c, _, _ = x.shape
    return x.permute(0, 2, 3, 1).view(b, -1, c).contiguous()


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C).contiguous()
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C).contiguous()
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1).contiguous()
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).contiguous()
    return x

class AFSIAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., use_relative_pe=False):

        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.frequency_weight = nn.Parameter(torch.tensor(0.5))
        self.use_relative_pe = use_relative_pe
        if self.use_relative_pe:
            window_size = (4, 4)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HSLFPN= nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_heads)
        )
        self.softmax = nn.Softmax(dim=-1)

    def extract_dct_features(self, x):
        B_, N, C = x.shape

        x_mean = x.mean(dim=-1)

        dct_features = torch.zeros_like(x_mean)

        for b in range(B_):
            feat_vec = x_mean[b].detach().cpu().numpy()

            feat_size = int(np.ceil(np.sqrt(N)))
            padded_feat = np.zeros((feat_size, feat_size))
            padded_feat.flat[:N] = feat_vec

            try:
                dct_coeffs = cv2.dct(np.float32(padded_feat))

                dct_flat = dct_coeffs.flat[:N]

                dct_features[b] = torch.from_numpy(dct_flat).to(x.device)
            except Exception as e:
                dct_features[b] = x_mean[b]

        dct_features = torch.clamp(dct_features, -10.0, 10.0)

        dct_norm = torch.norm(dct_features, p=2, dim=1, keepdim=True) + 1e-5
        dct_features = dct_features / dct_norm

        return dct_features

    def compute_frequency_weights(self, x):
        """
        Compute attention weights based on frequency domain features
        """
        try:
            dct_features = self.extract_dct_features(x)  # [B_, N]

            x_avg = x.mean(dim=1)  # [B_, C]
            head_weights = self.HSLFPN(x_avg)  # [B_, num_heads]

            N = x.size(1)
            dct_features = dct_features.unsqueeze(1)  # [B_, 1, N]
            dct_features = dct_features.repeat(1, self.num_heads, 1)  # [B_, num_heads, N]

            head_weights = head_weights.unsqueeze(-1)  # [B_, num_heads, 1]
            weighted_dct = dct_features * head_weights  # [B_, num_heads, N]

            freq_attn = torch.bmm(weighted_dct.transpose(1, 2), weighted_dct)  # [B_, N, N]

            sum_attn = freq_attn.sum(dim=-1, keepdim=True)
            sum_attn = torch.max(sum_attn, torch.ones_like(sum_attn) * 1e-5)
            freq_attn = freq_attn / sum_attn

            freq_attn = torch.clamp(freq_attn, 0.0, 1.0)

            return freq_attn.unsqueeze(1)

            return freq_attn.unsqueeze(1)  # [B_, 1, N, N]
        except Exception as e:
            B_, N = x.size(0), x.size(1)
            return torch.ones(B_, 1, N, N, device=x.device)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.use_relative_pe and N == 16:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                16, 16, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, 16, 16
            attn = attn + relative_position_bias.unsqueeze(0)

        if self.frequency_weight > 0:
            try:
                freq_attn = self.compute_frequency_weights(x)  # [B_, 1, N, N]

                freq_attn = freq_attn.repeat(1, self.num_heads, 1, 1)  # [B_, num_heads, N, N]

                adaptive_weight = torch.sigmoid(self.frequency_weight)

                combined_attn = (1 - adaptive_weight) * attn + adaptive_weight * freq_attn
                attn = combined_attn
            except Exception as e:
                pass
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()

        self.dwconv1 = nn.Conv2d(in_features, in_features // 2, 1, 1)
        self.dwconv2 = nn.Conv2d(in_features, in_features // 4, kernel_size=3, stride=1, padding=1)
        self.dwconv3 = nn.Conv2d(in_features, in_features // 4, kernel_size=7, stride=1, padding=3)

        self.act = act_layer()

        self.fc1 = nn.Conv2d(in_features, in_features * 4, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_features * 4, in_features, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = blc2bchw(x, H, W)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = bchw2blc(x, H, W)
        return x

class AFSIFormer(nn.Module):
    def __init__(self, dim, heads):
        super(AFSIFormer, self).__init__()

        self.dim = dim
        self.reduced_channel = 2

        self.down = nn.Sequential(
            nn.Conv2d(dim, dim // self.reduced_channel, 2, 2, 0, groups=dim // 8),
            nn.BatchNorm2d(dim // self.reduced_channel))

        self.h_conv = nn.Conv2d(dim // self.reduced_channel, dim // self.reduced_channel, 2, 2, 0)
        self.w_conv = nn.Conv2d(dim // self.reduced_channel, dim // self.reduced_channel, 2, 2, 0)

        self.BADAM = AFSIAttention(
            dim=dim // self.reduced_channel,
            num_heads=heads,
        )
        self.LWAM = AFSIAttention(
            dim=dim // self.reduced_channel,
            num_heads=heads,
            use_relative_pe=True,
        )

        self.conv_out = nn.Conv2d(dim // self.reduced_channel, dim, 3, 1, 1, groups=dim // 8)

        self.mlp = Mlp(in_features=dim // self.reduced_channel, act_layer=nn.GELU, drop=0.1)
        self.norm = nn.LayerNorm(dim // self.reduced_channel)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def borderindex(self, real_h):
        index = []
        windows = real_h // 4
        for i in range(windows):
            if i == 0:
                index.append(4 - 1)
            elif i == windows - 1:
                index.append(real_h - 4)
            else:
                index.append(i * 4)
                index.append(i * 4 + 3)
        return index

    def forward(self, x):
        x_down = self.down(x)
        x_down = x_down.permute(0, 2, 3, 1).contiguous()

        H, W = x_down.shape[1], x_down.shape[2]
        pad_r = int(((4 - W % 4) % 4) / 2)
        pad_b = pad_l = pad_t = pad_r
        if pad_r > 0:
            x_down = F.pad(x_down, (0, 0, pad_l, pad_r, pad_t, pad_b), mode='reflect')

        border_index = torch.Tensor(self.borderindex(x_down.shape[2])).int().to(x_down.device)

        x_h = torch.index_select(x_down, 1, border_index).permute(0, 3, 1, 2).contiguous()
        x_h = self.h_conv(x_h).permute(0, 2, 3, 1).contiguous()

        b_, h_, w_, c_ = x_h.shape
        x_h = window_partition(x_h, [1, w_]).view(-1, 1 * w_, c_).contiguous()

        x_w = torch.index_select(x_down, 2, border_index).permute(0, 3, 2, 1).contiguous()
        x_w = self.w_conv(x_w).permute(0, 2, 3, 1).contiguous()


        x_w = window_partition(x_w, [1, w_]).view(-1, 1 * w_, c_).contiguous()

        x_total = torch.cat([x_h, x_w], dim=0)

        x_h, x_w = torch.chunk(self.BADAM(x_total), 2, 0)
        x_h, x_w = x_h.contiguous(), x_w.contiguous()
        x_h, x_w = window_reverse(x_h, [1, w_], h_, w_).permute(0, 3, 1, 2).contiguous(), window_reverse(x_w, [1, w_],
                                                                                                         h_,
                                                                                                         w_).permute(0,
                                                                                                                     3,
                                                                                                                     2,
                                                                                                                     1).contiguous()
        x_h, x_w = F.interpolate(x_h, scale_factor=2, mode='bilinear', align_corners=True), F.interpolate(x_w,
                                                                                                          scale_factor=2,
                                                                                                          mode='bilinear',
                                                                                                          align_corners=True)
        x_h, x_w = x_h.permute(0, 2, 3, 1).contiguous(), x_w.permute(0, 2, 3,
                                                                     1).contiguous()

        x_down.index_add_(1, border_index, x_h)
        x_down.index_add_(2, border_index, x_w)

        lwam_local = window_partition(x_down, [4, 4]).view(-1, 16, x_down.shape[3]).contiguous()
        lwam_local = self.LWAM(lwam_local)

        lwam_local = window_reverse(lwam_local, [4, 4], x_down.shape[1],
                                       x_down.shape[2]).contiguous()

        if pad_r > 0:
            x_down = x_down[:, pad_t:H + pad_t, pad_l:W + pad_t, :].contiguous()
            lwam_local = lwam_local[:, pad_t:H + pad_t, pad_l:W + pad_t, :].contiguous()

        bb, hh, ww, cc = lwam_local.shape
        lwam_local = lwam_local.view(bb, hh * ww, cc).contiguous()

        lwam_local = self.mlp(self.norm(lwam_local), hh, ww)
        lwam_local = lwam_local.view(bb, hh, ww, cc).contiguous() + x_down
        lwam_local = lwam_local.permute(0, 3, 1, 2).contiguous()
        output = F.interpolate(lwam_local, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.conv_out(output)
        return output


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.a = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                              requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.a is not None:
            x = self.a * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):


    def __init__(self, in_chans=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.1,
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.former = nn.ModuleList([
            AFSIFormer(dims[0], 8),
            AFSIFormer(dims[1], 8),
            AFSIFormer(dims[2], 8),
            AFSIFormer(dims[3], 8),
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.former[i](x) + x
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
}

def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


class upsample(nn.Module):
    def __init__(self, in_c, factor):
        super(upsample, self).__init__()

        self.up_factor = factor
        self.factor1 = factor * factor // 2
        self.factor2 = factor * factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1 * in_c, (1, 7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1 * in_c, self.factor2 * in_c, (7, 1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor),
            nn.Conv2d(in_c, in_c, 3, groups=in_c // 4, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            upsample(ch_in, 2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



from bisem import BiSEM
class SynFSNet(nn.Module):
    def __init__(self, n_class=6, pretrained=True):
        super(SynFSNet, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        config = [96, 192, 384, 768]
        self.backbone = convnext_small(pretrained, True)

        self.former = nn.ModuleList([
            AFSIFormer(config[2], 8),
            AFSIFormer(config[1], 8),
            AFSIFormer(config[0], 8),
        ])

        self.Up5 = up_conv(ch_in=config[3], ch_out=config[3] // 2)
        self.Up_conv5 = conv_block(ch_in=config[3], ch_out=config[3] // 2)

        self.Up4 = up_conv(ch_in=config[2], ch_out=config[2] // 2)
        self.Up_conv4 = conv_block(ch_in=config[2], ch_out=config[2] // 2)

        self.Up3 = up_conv(ch_in=config[1], ch_out=config[1] // 2)
        self.Up_conv3 = conv_block(ch_in=config[1], ch_out=config[1] // 2)

        self.Up4x = upsample(config[0], 4)
        self.seghead = nn.Conv2d(config[0], n_class, kernel_size=1, stride=1, padding=0)
        self.BiSEM = BiSEM(
            in_channels=config[0],
            out_channels=config[0],
            height=128
        )


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        d3 = self.Up5(x4)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.Up_conv5(d3)
        d3 = self.former[0](d3) + d3
        d2 = self.Up4(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.Up_conv4(d2)
        d2 = self.former[1](d2) + d2
        d1 = self.Up3(d2)
        x1 = self.BiSEM(x1)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.Up_conv3(d1)
        d1 = self.former[2](d1) + d1
        d5 = self.Up4x(d1)
        out = self.seghead(d5)
        return out


# if __name__ == "__main__":
#
#     model = SynFSNet(6, False)
#     img = torch.rand((1, 3, 512, 512))
#     output = model(img)
#     print(output.shape)
#
#     if 1:
#         from fvcore.nn import FlopCountAnalysis, parameter_count_table
#
#         flops = FlopCountAnalysis(model, img)
#         print("FLOPs: %.4f G" % (flops.total() / 1e9))
#
#         total_paramters = 0
#         for parameter in model.parameters():
#             i = len(parameter.size())
#             p = 1
#             for j in range(i):
#                 p *= parameter.size(j)
#             total_paramters += p
#         print("Params: %.4f M" % (total_paramters / 1e6))

    """
    FLOPs: 73.5629 G
    Params: 68.4375 M
    """

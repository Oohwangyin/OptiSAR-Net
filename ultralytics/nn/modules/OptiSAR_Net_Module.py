import torch
import torch.nn as nn

from .conv import Conv

__all__ = (
    "DAAM",
    "DAAMChannelAttention",
)


class DAAMChannelAttention(nn.Module):
    """Use the DAAM semantic feature to modulate a detection feature for classification."""

    def __init__(self, daam_channels, target_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(daam_channels, target_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.constant_(self.conv.bias, 1.0)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, daam_feat, target_feat):
        weight = self.sigmoid(self.conv(self.gap(daam_feat)))
        return target_feat + target_feat * weight


class EnhancedConvolutionalBlock(nn.Module):
    """Lightweight feed-forward convolution block used inside DAAM."""

    def __init__(self, input_channels, hidden_channels=None, output_channels=None, dropout_rate=0.0, use_residual=True):
        super().__init__()
        output_channels = output_channels or input_channels
        hidden_channels = hidden_channels or input_channels

        self.conv_reduce = Conv(input_channels, hidden_channels, k=1)
        self.conv_expand = Conv(hidden_channels, output_channels, k=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_residual = use_residual

    def forward(self, x):
        residual = x
        x = self.dropout(self.conv_reduce(x))
        x = self.dropout(self.conv_expand(x))
        return x + residual if self.use_residual else x


class DualAdaptiveAttention(nn.Module):
    """Dual local/global adaptive attention used by DAAM."""

    def __init__(self, dim):
        super().__init__()
        self.input_proj = Conv(dim, dim, k=1)
        self.activation = nn.GELU()
        self.output_proj = Conv(dim, dim, k=1)

        self.local_conv = Conv(dim, dim, k=3, p=1, g=dim)
        self.global_conv = Conv(dim, dim, k=3, p=3, g=dim, d=3)

        self.channel_reducer_local = Conv(dim, dim // 2, k=1)
        self.channel_reducer_global = Conv(dim, dim // 2, k=1)
        self.attention_squeeze = Conv(2, 2, k=7, p=3)
        self.channel_mixer = Conv(dim // 2, dim, k=1)

    def forward(self, x):
        residual = x
        x = self.activation(self.input_proj(x))

        local_features = self.local_conv(x)
        global_features = self.global_conv(local_features)

        attn_local = self.channel_reducer_local(local_features)
        attn_global = self.channel_reducer_global(global_features)
        attn_combined = torch.cat([attn_local, attn_global], dim=1)

        attn_avg = torch.mean(attn_combined, dim=1, keepdim=True)
        attn_max, _ = torch.max(attn_combined, dim=1, keepdim=True)
        attn_weights = self.attention_squeeze(torch.cat([attn_avg, attn_max], dim=1)).sigmoid()

        weighted_attn = (
            attn_local * attn_weights[:, 0, :, :].unsqueeze(1)
            + attn_global * attn_weights[:, 1, :, :].unsqueeze(1)
        )

        x = global_features * self.channel_mixer(weighted_attn)
        return self.output_proj(x) + residual


class DAAM(nn.Module):
    """Dual Attention Adaptive Module."""

    def __init__(self, dim, use_auto_layer_scaling=True, layer_scale_init_value=1e-2):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.daa = DualAdaptiveAttention(dim)
        self.ecb = EnhancedConvolutionalBlock(dim, dim)

        self.use_auto_layer_scaling = use_auto_layer_scaling
        if use_auto_layer_scaling:
            self.layer_scale_daa = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_ecb = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.layer_scale_daa = None
            self.layer_scale_ecb = None

    def forward(self, x):
        attention_output = self.daa(self.norm1(x))
        if self.use_auto_layer_scaling:
            attention_output = self.layer_scale_daa.unsqueeze(-1).unsqueeze(-1) * attention_output
        x = x + attention_output

        ecb_output = self.ecb(self.norm2(x))
        if self.use_auto_layer_scaling:
            ecb_output = self.layer_scale_ecb.unsqueeze(-1).unsqueeze(-1) * ecb_output
        return x + ecb_output

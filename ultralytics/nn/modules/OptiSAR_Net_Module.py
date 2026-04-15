import einops
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.parameter import Parameter
from .conv import Conv, autopad

__all__ = (
    "DAAM",
    "BSPPF",
    "VSSA",
    "CAAM",
    "LKA_SPPF",
    "VCAA",
    "C2f_DFDA",
    "CSAF",
    "FPM"
)

class EnhancedConvolutionalBlock(nn.Module):
    """
    Enhanced Convolutional Block (ECB) that applies two 1x1 convolutions with optional residual connection.
    """

    def __init__(self, input_channels, hidden_channels=None, output_channels=None,
                 dropout_rate=0., use_residual=True):
        super().__init__()

        # If not specified, keep the number of channels constant
        output_channels = output_channels or input_channels
        hidden_channels = hidden_channels or input_channels

        # First 1x1 convolution layer
        self.conv_reduce = Conv(input_channels, hidden_channels, k=1)

        # Second 1x1 convolution layer
        self.conv_expand = Conv(hidden_channels, output_channels, k=1)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Flag to control residual connection
        self.use_residual = use_residual

    def forward(self, x):
        # Store the input for potential residual connection
        residual = x

        # First convolution
        x = self.conv_reduce(x)
        x = self.dropout(x)

        # Second convolution
        x = self.conv_expand(x)
        x = self.dropout(x)

        # Add residual connection if enabled
        if self.use_residual:
            return x + residual
        else:
            return x


class DualAdaptiveAttention(nn.Module):
    """
    Dual Adaptive Attention (DAA).
    """
    def __init__(self, dim):
        super().__init__()

        # Input projection
        self.input_proj = Conv(dim, dim, k=1)
        self.activation = nn.GELU()

        # Output projection
        self.output_proj = Conv(dim, dim, k=1)

        # Depth-wise convolutions for local and global feature extraction
        self.local_conv = Conv(dim, dim, k=3, p=2, g=dim, d=2)
        # 原代码
        # self.global_conv = Conv(dim, dim, k=3, p=3, g=dim, d=3)

        # 修改后：d=5，增强对飞机长距离几何结构（如机身轴线）的感知
        self.global_conv = Conv(dim, dim, k=3, p=4, g=dim, d=4)

        # Channel reduction for attention computation
        self.channel_reducer_local = Conv(dim, dim // 2, k=1)
        self.channel_reducer_global = Conv(dim, dim // 2, k=1)

        # Attention squeeze operation
        self.attention_squeeze = Conv(2, 2, k=7, p=3)

        # Final channel mixing
        self.channel_mixer = Conv(dim // 2, dim, k=1)

    def forward(self, x):
        # Store input for residual connection
        residual = x.clone()

        # Input projection and activation
        x = self.input_proj(x)
        x = self.activation(x)

        # Local feature extraction
        local_features = self.local_conv(x)

        # Global feature extraction
        global_features = self.global_conv(local_features)

        # Compute attention for local and global features
        attn_local = self.channel_reducer_local(local_features)
        attn_global = self.channel_reducer_global(global_features)

        # Concatenate local and global attention
        attn_combined = torch.cat([attn_local, attn_global], dim=1)

        # Compute average and max attention
        attn_avg = torch.mean(attn_combined, dim=1, keepdim=True)
        attn_max, _ = torch.max(attn_combined, dim=1, keepdim=True)

        # Aggregate average and max attention
        attn_pooled = torch.cat([attn_avg, attn_max], dim=1)

        # Apply attention squeeze and sigmoid activation
        attn_weights = self.attention_squeeze(attn_pooled).sigmoid()

        # Compute weighted attention
        weighted_attn = (
            attn_local * attn_weights[:, 0, :, :].unsqueeze(1) +
            attn_global * attn_weights[:, 1, :, :].unsqueeze(1)
        )

        # Mix channels in the attention
        attn_mixed = self.channel_mixer(weighted_attn)

        # Apply attention to global features
        x = global_features * attn_mixed

        # Output projection
        x = self.output_proj(x)

        # Add residual connection
        x = x + residual

        return x


class DAAM(nn.Module):
    """
    双层级路由可变性空间金字塔池化
    Dual Attention Adaptive Module (DAAM) that combines attention and ECB
    with optional layer scaling.
    """

    def __init__(self, dim, use_auto_layer_scaling=True, layer_scale_init_value=1e-2):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.daa = DualAdaptiveAttention(dim)
        self.ecb = EnhancedConvolutionalBlock(dim, dim)

        self.use_auto_layer_scaling = use_auto_layer_scaling
        if use_auto_layer_scaling:
            self.layer_scale_daa = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_ecb = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale_daa = None
            self.layer_scale_ecb = None

    def forward(self, x):
        # Apply attention
        attention_output = self.daa(self.norm1(x))
        if self.use_auto_layer_scaling:
            attention_output = self.layer_scale_daa.unsqueeze(-1).unsqueeze(-1) * attention_output
        x = x + attention_output

        # Apply ECB
        ecb_output = self.ecb(self.norm2(x))
        if self.use_auto_layer_scaling:
            ecb_output = self.layer_scale_ecb.unsqueeze(-1).unsqueeze(-1) * ecb_output
        x = x + ecb_output

        return x


class TopkRouting(nn.Module):
    """
    Differentiable top-k routing with scaling, adapted from bi-level routing attention.
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """

        query, key = query.detach(), key.detach()
        attn_logit = (query * self.scale) @ key.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    """
       KVGather module for efficient key-value pair selection based on routing indices.
       This module is part of the bi-level routing attention mechanism.
    """

    def __init__(self):
        super().__init__()

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)
        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        return topk_kv


class BiLevelRoutingDeformableAttention(nn.Module):
    """
    Bi-Level Routing Deformable Attention module.

    This module combines local attention with global routing and deformable convolutions
    for enhanced feature extraction in computer vision tasks.
    """

    def __init__(self, dim, num_windows=11, num_heads=4, qk_dim=None, qk_scale=None,
                 kv_per_window=4, kv_downsample_mode='ada_maxpool', topk=4,
                 side_conv=3, use_deformable=True, off_conv=9
                ):
        """
        Initialize the Bi-Level Routing Deformable Attention.

        Args:
            dim (int): Number of input channels.
            num_windows (int): Number of windows in each dimension for local attention.
            num_heads (int): Number of attention heads.
            qk_dim (int): Dimension of query and key vectors. If None, set to dim. Default is None.
            qk_scale (float): Scaling factor for query-key dot product. If None, set to 1/sqrt(qk_dim).
            kv_per_window (int): Number of key-value pairs per window for downsampling.
            kv_downsample_mode (str): Mode for downsampling key-value pairs. Options: 'ada_avgpool', 'ada_maxpool'.
            topk (int): Number of top attention scores to consider in routing.
            side_dwconv (int): Kernel size for depthwise convolution in LEPE. Set to 0 to disable.
            use_deformable (bool):
            auto_pad (bool): Whether to automatically pad input to match window size.
        """
        super().__init__()
        self.dim = dim
        self.num_windows = num_windows
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        # Side-enhanced convolution (SEC)
        self.sec = nn.Conv2d(dim, dim, kernel_size=side_conv, stride=1, padding=side_conv // 2,
                              groups=dim) if side_conv > 0 else \
            lambda x: torch.zeros_like(x)

        # Global routing settings
        self.topk = topk
        self.router = TopkRouting(qk_dim=self.qk_dim, qk_scale=self.scale, topk=self.topk)
        self.kv_gather = KVGather()

        # Query, Key, Value projections
        self.query_proj = Conv(dim, dim, 1)
        self.kv_proj = Conv(dim, dim * 2, 1)

        # Key-Value downsampling
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_window = kv_per_window
        if self.kv_downsample_mode == 'ada_avgpool':
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_window)
        elif self.kv_downsample_mode == 'ada_maxpool':
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_window)
        else:
            self.kv_down = nn.Identity()

        self.attn_act = nn.Softmax(dim=-1)
        self.use_deformable = use_deformable
        self.off_conv = off_conv

        # Offset prediction
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=self.off_conv, stride=1, padding=self.off_conv // 2, groups=self.dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(self.dim, 2, kernel_size=1, stride=1, padding=0, bias=False)
        )



    @torch.no_grad()
    def _get_reference_points(self, height, width, batch_size, dtype, device):
        """
        Generate reference points for deformable attention.

        Args:
            height (int): Height of the feature map.
            width (int): Width of the feature map.
            batch_size (int): Batch size.
            dtype (torch.dtype): Data type of the tensor.
            device (torch.device): Device to create the tensor on.

        Returns:
            torch.Tensor: Reference points of shape (batch_size, height, width, 2).
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(width - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(height - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(batch_size, -1, -1, -1)
        return ref

    def forward(self, x):
        x = rearrange(x, "n c h w -> n h w c")

        # Auto-padding
        batch_size, height_in, width_in, channels = x.size()
        pad_left = pad_top = 0
        pad_right = (self.num_windows - width_in % self.num_windows) % self.num_windows
        pad_bottom = (self.num_windows - height_in % self.num_windows) % self.num_windows
        x = F.pad(x, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
        _, height, width, _ = x.size()  # padded size

        # Reshape input for window-based processing
        x = rearrange(x, "n (j h) (i w) c -> n c (j h) (i w)", j=self.num_windows, i=self.num_windows)

        # Query projection
        query = self.query_proj(x)
        query_offset = query
        query = rearrange(query, "n c (j h) (i w) -> n (j i) h w c", j=self.num_windows, i=self.num_windows)

        # Deformable offset calculation
        offset = self.offset_predictor(query_offset).contiguous()
        height_key, width_key = offset.size(2), offset.size(3)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        dtype, device = x.dtype, x.device
        batch_size, _, _, _ = offset.size()
        reference = self._get_reference_points(height_key, width_key, batch_size, dtype, device)

        # 原代码：约束极其严格，只允许 5% 的偏移，适合海面小目标船舶
        # pos = (offset + reference).clamp(-0.05, +0.05)
        # 修改后：允许 20% 的偏移量，使采样点能跳出局部，覆盖到延展的机翼和机身
        pos = (offset + reference).clamp(-0.12, +0.12)
        # Apply deformable sampling
        if self.use_deformable:
            x_sampled = F.grid_sample(
                input=x,
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)
        else:
            x_sampled = x

        # Key-Value projection
        kv = self.kv_proj(x_sampled)
        kv = rearrange(kv, "n c (j h) (i w) -> n (j i) h w c", j=self.num_windows, i=self.num_windows)


        # Reshape for pixel-wise and window-wise operations
        query_pixel = rearrange(query, 'n p2 h w c -> n p2 (h w) c')
        kv_pixel = rearrange(kv, 'n p2 h w c -> (n p2) c h w')

        # Downsample key-value pairs
        kv_pixel = self.kv_down(kv_pixel)
        kv_pixel = rearrange(kv_pixel, '(n j i) c h w -> n (j i) (h w) c', j=self.num_windows, i=self.num_windows)

        # Window-wise query and key
        query_window, key_window = query.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])

        # Side-enhanced convolution (SEC)
        sec = self.sec(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.num_windows,
                                   i=self.num_windows).contiguous())
        sec = rearrange(sec, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.num_windows, i=self.num_windows)

        # Global routing
        routing_weights, routing_indices = self.router(query_window, key_window)

        # Gather key-value pairs based on routing
        kv_pixel_selected = self.kv_gather(r_idx=routing_indices, r_weight=routing_weights, kv=kv_pixel)
        key_pixel_selected, value_pixel_selected = kv_pixel_selected.split([self.qk_dim, self.dim], dim=-1)

        # Reshape for multi-head attention
        key_pixel_selected = rearrange(key_pixel_selected, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        value_pixel_selected = rearrange(value_pixel_selected, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        query_pixel = rearrange(query_pixel, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)

        # Compute attention weights and apply attention
        attn_weights = (query_pixel * self.scale) @ key_pixel_selected
        attn_weights = self.attn_act(attn_weights)
        out = attn_weights @ value_pixel_selected

        # Reshape output and add SEC
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.num_windows, i=self.num_windows,
                        h=height // self.num_windows, w=width // self.num_windows)
        out = out + sec

        # Remove padding if applied
        out = out[:, :height_in, :width_in, :].contiguous()

        return rearrange(out, "n h w c -> n c h w")


class BSPPF(nn.Module):
    # The Bi-Level Routing Deformable Spatial Pyramid Pooling - Fast (BSPPF) layer
    # is used to dynamically adjust the allocation of multi-scale feature space.

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # 原代码
        # self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 兼容 YAML 传入的列表参数，取第一个值 5 作为基础 kernel
        if isinstance(k, list):
            k = k[0]

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.brda = BiLevelRoutingDeformableAttention(c_)

        # [新增]：引入动态尺度加权模块，输入通道数是拼接后的 c_ * 4
        # self.scale_weighting = DynamicScaleWeighting(c_ * 4)

    def forward(self, x):
        x = self.cv1(x)
        x = x + self.brda(x)

        # 混合池化策略：前两层用 Max 抓取边缘与局部散射点，最后一层用 Avg 抑制宏观噪声
        y1 = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        y2 = F.max_pool2d(y1, kernel_size=5, stride=1, padding=2)

        # 【关键修改】宏观分支使用平均池化，防止 SAR 背景强噪点被传递
        y3 = F.avg_pool2d(y2, kernel_size=5, stride=1, padding=2)

        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class SpatialShuffleAttention(nn.Module):
    """
    Spatial Shuffle Attention (SSA).
    """

    def __init__(self, dim, groups=8, dropout_rate=0.1):
        """
        Initialize the Spatial Shuffle Attention.

        Args:
            dim (int): Number of input channels. Default is 512.
            groups (int): Number of groups for channel shuffling. Default is 8.
            dropout_rate (float): Dropout rate for regularization. Default is 0.1.
        """
        super().__init__()
        # self.groups = groups
        self.groups = max(2, groups // 2)
        self.dim = dim

        # Pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Learnable parameters for spatial attention
        self.weight_max = Parameter(torch.zeros(1, dim // (2 * self.groups), 1, 1))
        self.bias_max = Parameter(torch.ones(1, dim // (2 * self.groups), 1, 1))
        self.weight_avg = Parameter(torch.zeros(1, dim // (2 * self.groups), 1, 1))
        self.bias_avg = Parameter(torch.ones(1, dim // (2 * self.groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def channel_shuffle(x, groups):
        """
        Perform channel shuffling operation.

        Args:
            x (torch.Tensor): Input tensor.
            groups (int): Number of groups for shuffling.

        Returns:
            torch.Tensor: Channel shuffled tensor.
        """
        batch_size, channels, height, width = x.shape
        channels_per_group = channels // groups

        # Reshape and transpose for shuffling
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()

        # Flatten the grouped channels
        x = x.view(batch_size, -1, height, width)

        return x

    def forward(self, x):
        b, c, h, w = x.size()

        # Group the input into subfeatures
        x = x.view(b * self.groups, -1, h, w)  # (b*groups, c//groups, h, w)

        # Apply initial channel shuffle
        x = self.channel_shuffle(x, 2)

        # Partitioning subspace
        x_1, x_2 = x.chunk(2, dim=1)

        # Apply pooling operations
        avg_pool = self.avg_pool(x_1)  # (batch_size*groups, channels//(2*groups), 1, 1)
        max_pool = self.max_pool(x_2)  # (batch_size*groups, channels//(2*groups), 1, 1)

        # Embedding global and key information
        avg_attention = self.weight_avg * avg_pool + self.bias_avg
        max_attention = self.weight_max * max_pool + self.bias_max

        # Dual-path spatial attention fusion
        channel_attention = torch.cat((max_attention, avg_attention), dim=1)
        channel_attention = self.sigmoid(channel_attention)

        # Apply attention and dropout
        x = x * self.dropout(channel_attention)

        # Reshape back to original dimensions
        out = x.contiguous().view(b, -1, h, w)

        # Final channel shuffle
        out = self.channel_shuffle(out, 2)

        return out


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, 1, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 3, 1),
            GSConv(c_, c2, 3, 1, act=False))

        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VSSA(nn.Module):
    """
    VoVGSCSP module with Spatial Shuffle Attention (VSSA).
    """

    def __init__(self, in_channels, out_channels, num_gsb=4, expansion_factor=0.5, dropout_rate=0.01):
        """
        Initialize the VSSA module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_gsb (int): Number of GSBottleneck layers. Default is 4.
            expansion_factor (float): Factor to determine the number of hidden channels. Default is 0.5.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion_factor)  # Calculate hidden channels
        self.conv_input_1 = Conv(in_channels, hidden_channels, k=1, s=1)
        self.conv_input_2 = Conv(in_channels, hidden_channels, k=1, s=1)

        # GSBottleneck sequence
        self.gsb_sequence = nn.Sequential(
            *(GSBottleneck(hidden_channels, hidden_channels, e=1.0) for _ in range(num_gsb))
        )
        self.conv_output = Conv(2 * hidden_channels, out_channels, k=1)

        # Spatial Shuffle Attention
        self.spatial_shuffle_attention = SpatialShuffleAttention(in_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply Spatial Shuffle Attention and dropout
        attended_features = self.dropout(self.spatial_shuffle_attention(x))

        # Process through GSBottleneck sequence
        gsb_output = self.gsb_sequence(self.conv_input_1(attended_features))

        # Direct path through second input convolution
        direct_path = self.conv_input_2(attended_features)

        # Concatenate and process through output convolution
        combined_features = torch.cat((direct_path, gsb_output), dim=1)
        output = self.conv_output(combined_features)

        return output

class CrossAxialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用非对称卷积捕捉水平（机翼）和垂直（机身）特征
        self.conv_1x7 = Conv(dim, dim, k=(1, 7), p=(0, 3), g=dim)
        self.conv_7x1 = Conv(dim, dim, k=(7, 1), p=(3, 0), g=dim)
        self.conv_proj = Conv(dim, dim, k=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        # 十字形特征聚合
        attn = self.conv_1x7(x) + self.conv_7x1(x)
        attn = self.conv_proj(attn)
        return u * self.act(attn)

class CAAM(nn.Module):
    """
    专门针对飞机十字形拓扑结构的交叉轴向自适应模块
    替代原有的 DAAM
    """
    def __init__(self, c1, c2):
        super().__init__()
        dim = c1  # 保证内部维度一致
        self.norm = nn.BatchNorm2d(dim)
        self.caa = CrossAxialAttention(dim)
        self.ecb = EnhancedConvolutionalBlock(dim, dim)

    def forward(self, x):
        # 交叉轴向注意力注入
        x = x + self.caa(self.norm(x))
        # 特征增强
        x = x + self.ecb(x)
        return x


class LargeKernelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 局部结构提取 (相当于机身)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 极大感受野上下文 (dilation=3的7x7卷积，覆盖极广，用于包裹整架大飞机和离散SAR散斑)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class LKA_SPPF(nn.Module):
    """
    大核注意力池化模块，替代原生船舶 BSPPF 中不稳定的路由机制
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # 引入极其稳定的 LKA 替代 BiLevelRouting
        self.lka = LargeKernelAttention(c_)

    def forward(self, x):
        x = self.cv1(x)
        # LKA 重连离散特征
        x = x + self.lka(x)

        # 经典的 SPPF 级联池化，稳定可靠
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # 沿机身和机翼(X/Y轴)分别池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # 滤除斜向的机场建筑噪声
        out = identity * a_w * a_h
        return out


class VCAA(nn.Module):
    """
    坐标注意力增强的 VoV-GSCSP，替代原有通道洗牌的 VSSA
    """

    def __init__(self, in_channels, out_channels, num_gsb=4, expansion_factor=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion_factor)
        self.conv_input_1 = Conv(in_channels, hidden_channels, k=1, s=1)
        self.conv_input_2 = Conv(in_channels, hidden_channels, k=1, s=1)

        self.gsb_sequence = nn.Sequential(
            *(GSBottleneck(hidden_channels, hidden_channels, e=1.0) for _ in range(num_gsb))
        )
        self.conv_output = Conv(2 * hidden_channels, out_channels, k=1)

        # 引入坐标注意力，完美滤除背景
        self.coord_attn = CoordinateAttention(in_channels)

    def forward(self, x):
        attended_features = self.coord_attn(x)
        gsb_output = self.gsb_sequence(self.conv_input_1(attended_features))
        direct_path = self.conv_input_2(attended_features)
        combined_features = torch.cat((direct_path, gsb_output), dim=1)
        output = self.conv_output(combined_features)
        return output


class DFDA(nn.Module):
    """
    双频解耦注意力模块 (Dual-Frequency Decoupling Attention)
    """

    # 【修改这里】：将 dim 改为 c1, c2，以匹配 YOLO 的解析机制
    def __init__(self, c1, c2):
        super().__init__()
        dim = c1  # 在内部将 c1 赋值给 dim，确保后续逻辑正常运行

        # 低频分支 (光学轮廓)
        self.low_freq_path = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(dim, dim // 2, k=3, p=1)
        )
        # 高频分支 (SAR 散斑)
        self.high_freq_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv(dim, dim // 2, k=3, d=2, p=2)
        )
        self.fusion_gate = nn.Sequential(
            Conv(dim, dim, k=1),
            nn.Sigmoid()
        )
        self.out_proj = Conv(dim, dim, k=1)

    def forward(self, x):
        feat_low = self.low_freq_path(x)
        feat_high = self.high_freq_path(x)
        feat_cat = torch.cat([feat_low, feat_high], dim=1)
        gate = self.fusion_gate(feat_cat)
        return x + self.out_proj(feat_cat * gate)


class DSA_SPPF(nn.Module):
    """
    离散散射空洞空间金字塔池化 (Discrete Scattering Atrous SPPF)
    利用空洞池化连接遥远且离散的 SAR 飞机散射点
    """

    def __init__(self, c1, c2, k=5):  # 保留 k=5 以吸收 YAML 传来的多余参数
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        # 1. 基础池化 (dilation=1, padding=1，满足 PyTorch 的要求)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1)

        # 2. 空洞池化 (dilation=2, 数学上需要 padding=2)
        # 绕开 PyTorch 限制：在外部用 ReplicationPad 手动填充边缘，池化层内部设为 padding=0
        self.pad2 = nn.ReplicationPad2d(2)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=2)

        # 3. 空洞池化 (dilation=3, 数学上需要 padding=3)
        self.pad3 = nn.ReplicationPad2d(3)
        self.m3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=3)

    def forward(self, x):
        x = self.cv1(x)

        # 级联空洞池化与手动填充
        y1 = self.m1(x)
        y2 = self.m2(self.pad2(y1))
        y3 = self.m3(self.pad3(y2))

        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class BOD(nn.Module):
    """纯粹的背景正交剥离注意力机制"""

    def __init__(self, dim):
        super().__init__()
        self.vertical_conv = Conv(dim, dim, k=(7, 1), p=(3, 0), g=dim)
        self.horizontal_conv = Conv(dim, dim, k=(1, 7), p=(0, 3), g=dim)
        self.fusion = Conv(dim, dim, k=1)
        self.attention = nn.Sigmoid()

    def forward(self, x):
        target_feat = self.vertical_conv(x) * self.horizontal_conv(x)
        attn_mask = self.attention(self.fusion(target_feat))
        return x * attn_mask + x


class VBOD(nn.Module):
    """
    完整替换原 VSSA 的架构模块：
    承担通道降维映射 (c1 -> c2) + n次循环瓶颈 (n=3) + BOD十字注意力
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        hidden_channels = int(c2 * e)
        self.conv_input_1 = Conv(c1, hidden_channels, k=1, s=1)
        self.conv_input_2 = Conv(c1, hidden_channels, k=1, s=1)

        # 内部处理 yaml 传来的 n=3 次重复
        self.gsb_sequence = nn.Sequential(
            *(GSBottleneck(hidden_channels, hidden_channels, e=1.0) for _ in range(n))
        )
        self.conv_output = Conv(2 * hidden_channels, c2, k=1)

        # 接入 BOD
        self.bod = BOD(c1)

    def forward(self, x):
        attended_features = self.bod(x)
        gsb_output = self.gsb_sequence(self.conv_input_1(attended_features))
        direct_path = self.conv_input_2(attended_features)
        combined_features = torch.cat((direct_path, gsb_output), dim=1)
        return self.conv_output(combined_features)

class Bottleneck_DFDA(nn.Module):
    """
    将双频解耦机制 (DFDA) 注入到标准瓶颈层中
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # 【核心突变】：在特征提取的中央嵌入 DFDA，持续保护高低频特征
        self.dfda = DFDA(c_, c_)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 带有跨模态解耦的残差连接
        return x + self.cv2(self.dfda(self.cv1(x))) if self.add else self.cv2(self.dfda(self.cv1(x)))

class C2f_DFDA(nn.Module):
    """
    双频解耦跨阶段局部网络 (Cross-Modal C2f)
    替换 Backbone 中的标准 C2f，这是带来巨大 mAP 提升的绝对主力
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 【替换】：内部循环使用带有 DFDA 的瓶颈层
        self.m = nn.ModuleList(Bottleneck_DFDA(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CSAF(nn.Module):
    """
    跨尺度对齐融合模块 (Cross-Scale Alignment Fusion)
    替换 Neck 阶段粗暴的 Concat 操作，对齐错位的多模态特征
    """

    # 【修复】：将参数修改为 d=1，并使用 *args 吸收掉任何多余的 YOLO 参数
    def __init__(self, d=1, *args, **kwargs):
        super().__init__()
        self.d = d  # 拼接的维度，通常传过来的是 1
        # 简单的空间对齐注意力
        self.align_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 是一个包含两个张量的列表，如 [特征图1, 特征图2]

        # 沿着指定维度（d=1）将两者拼接，兼容 YOLO 的原生逻辑
        f_cat = torch.cat(x, dim=self.d)

        # 提取跨尺度的空间极值，生成对齐掩码
        avg_out = torch.mean(f_cat, dim=1, keepdim=True)
        max_out, _ = torch.max(f_cat, dim=1, keepdim=True)
        align_mask = self.sigmoid(self.align_conv(torch.cat([avg_out, max_out], dim=1)))

        # 输出对齐后的融合特征
        return f_cat * align_mask

class FPM(nn.Module):
    """
    超轻量级特征纯化模块 (Feature Purification Module)
    插入在 Backbone 提取后、Neck 拼接前，用于过滤干扰噪声
    """
    def __init__(self, c1, c2=None): # 兼容 YOLO 参数解析
        super().__init__()
        c2 = c2 or c1
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(8, c1 // 4), 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(max(8, c1 // 4), c1, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        return self.conv(x * self.channel_att(x))
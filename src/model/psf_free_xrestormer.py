import numbers
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.nn import functional as F


def _tensor_options(value):
    return {"device": value.device, "dtype": value.dtype}


def _expand_dim(value, dim, size):
    value = value.unsqueeze(dim)
    shape = [-1] * value.ndim
    shape[dim] = size
    return value.expand(*shape)


def _relative_to_absolute(value):
    batch, length, relative_length = value.shape
    radius = (relative_length + 1) // 2
    column = torch.zeros((batch, length, 1), **_tensor_options(value))
    value = torch.cat((value, column), dim=2)
    value = rearrange(value, "b l c -> b (l c)")
    padding = torch.zeros((batch, relative_length - length), **_tensor_options(value))
    value = torch.cat((value, padding), dim=1)
    value = value.reshape(batch, length + 1, relative_length)
    return value[:, :length, -radius:]


def _relative_logits_1d(query, relative_key):
    batch, height, width, _ = query.shape
    radius = (relative_key.shape[0] + 1) // 2
    logits = einsum("b x y d, r d -> b x y r", query, relative_key)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = _relative_to_absolute(logits)
    logits = logits.reshape(batch, height, width, radius)
    return _expand_dim(logits, dim=2, size=radius)


class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        scale = dim_head**-0.5
        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(rel_size * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(rel_size * 2 - 1, dim_head) * scale)

    def forward(self, query):
        query = rearrange(
            query,
            "b (x y) c -> b x y c",
            x=self.block_size,
        )
        width_logits = _relative_logits_1d(query, self.rel_width)
        width_logits = rearrange(
            width_logits,
            "b x i y j -> b (x y) (i j)",
        )

        query = rearrange(query, "b x y d -> b y x d")
        height_logits = _relative_logits_1d(query, self.rel_height)
        height_logits = rearrange(
            height_logits,
            "b x i y j -> b (y x) (j i)",
        )
        return width_logits + height_logits


def _to_3d(value):
    return rearrange(value, "b c h w -> b (h w) c")


def _to_4d(value, height, width):
    return rearrange(value, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("normalized_shape must contain one dimension")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, value):
        variance = value.var(-1, keepdim=True, unbiased=False)
        return value / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("normalized_shape must contain one dimension")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, value):
        mean = value.mean(-1, keepdim=True)
        variance = value.var(-1, keepdim=True, unbiased=False)
        return (value - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError("layer_norm_type must be BiasFree or WithBias")

    def forward(self, value):
        height, width = value.shape[-2:]
        return _to_4d(self.body(_to_3d(value)), height, width)


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            3,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, value):
        first, second = self.dwconv(self.project_in(value)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            3,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, value):
        _, _, height, width = value.shape
        query, key, content = self.qkv_dwconv(self.qkv(value)).chunk(3, dim=1)
        query = rearrange(
            query,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        key = rearrange(
            key,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        content = rearrange(
            content,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (query @ key.transpose(-2, -1)) * self.temperature
        output = attention.softmax(dim=-1) @ content
        output = rearrange(
            output,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.project_out(output)


class OCAB(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        overlap_ratio,
        num_heads,
        dim_head,
        bias,
    ):
        super().__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = dim_head * num_heads
        self.scale = dim_head**-0.5
        padding = (self.overlap_win_size - window_size) // 2
        self.unfold = nn.Unfold(
            kernel_size=self.overlap_win_size,
            stride=window_size,
            padding=padding,
        )
        self.qkv = nn.Conv2d(dim, self.inner_dim * 3, 1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, 1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size=window_size,
            rel_size=self.overlap_win_size,
            dim_head=dim_head,
        )

    def forward(self, value):
        _, _, height, width = value.shape
        query, key, content = self.qkv(value).chunk(3, dim=1)
        query = rearrange(
            query,
            "b c (h p1) (w p2) -> (b h w) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        key = self.unfold(key)
        content = self.unfold(content)
        key = rearrange(
            key,
            "b (c j) i -> (b i) j c",
            c=self.inner_dim,
        )
        content = rearrange(
            content,
            "b (c j) i -> (b i) j c",
            c=self.inner_dim,
        )
        query = rearrange(
            query,
            "b n (head c) -> (b head) n c",
            head=self.num_spatial_heads,
        )
        key = rearrange(
            key,
            "b n (head c) -> (b head) n c",
            head=self.num_spatial_heads,
        )
        content = rearrange(
            content,
            "b n (head c) -> (b head) n c",
            head=self.num_spatial_heads,
        )
        query = query * self.scale
        attention = query @ key.transpose(-2, -1)
        attention = (attention + self.rel_pos_emb(query)).softmax(dim=-1)
        output = attention @ content
        output = rearrange(
            output,
            "(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)",
            head=self.num_spatial_heads,
            h=height // self.window_size,
            w=width // self.window_size,
            p1=self.window_size,
            p2=self.window_size,
        )
        return self.project_out(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        overlap_ratio,
        channel_heads,
        spatial_heads,
        spatial_dim_head,
        expansion_factor,
        bias,
        layer_norm_type,
    ):
        super().__init__()
        self.spatial_attn = OCAB(
            dim,
            window_size,
            overlap_ratio,
            spatial_heads,
            spatial_dim_head,
            bias,
        )
        self.channel_attn = ChannelAttention(dim, channel_heads, bias)
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.norm3 = LayerNorm(dim, layer_norm_type)
        self.norm4 = LayerNorm(dim, layer_norm_type)
        self.channel_ffn = FeedForward(dim, expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, expansion_factor, bias)

    def forward(self, value):
        value = value + self.channel_attn(self.norm1(value))
        value = value + self.channel_ffn(self.norm2(value))
        value = value + self.spatial_attn(self.norm3(value))
        return value + self.spatial_ffn(self.norm4(value))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, input_channels, dim, bias):
        super().__init__()
        self.proj = nn.Conv2d(input_channels, dim, 3, padding=1, bias=bias)

    def forward(self, value):
        return self.proj(value)


class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, value):
        return self.body(value)


class Upsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, value):
        return self.body(value)


def _blocks(
    count,
    dim,
    window_size,
    overlap_ratio,
    channel_heads,
    spatial_heads,
    spatial_dim_head,
    expansion_factor,
    bias,
    layer_norm_type,
):
    return nn.Sequential(
        *[
            TransformerBlock(
                dim,
                window_size,
                overlap_ratio,
                channel_heads,
                spatial_heads,
                spatial_dim_head,
                expansion_factor,
                bias,
                layer_norm_type,
            )
            for _ in range(count)
        ]
    )


class XRestormer(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        dim=48,
        num_blocks=(2, 4, 4, 4),
        num_refinement_blocks=4,
        channel_heads=(1, 2, 4, 8),
        spatial_heads=(1, 2, 4, 8),
        overlap_ratio=(0.5, 0.5, 0.5, 0.5),
        window_size=8,
        spatial_dim_head=16,
        expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(input_channels, dim, bias)
        self.encoder_level1 = _blocks(
            num_blocks[0],
            dim,
            window_size,
            overlap_ratio[0],
            channel_heads[0],
            spatial_heads[0],
            spatial_dim_head,
            expansion_factor,
            bias,
            layer_norm_type,
        )
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = _blocks(
            num_blocks[1],
            dim * 2,
            window_size,
            overlap_ratio[1],
            channel_heads[1],
            spatial_heads[1],
            spatial_dim_head,
            expansion_factor,
            bias,
            layer_norm_type,
        )
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = _blocks(
            num_blocks[2],
            dim * 4,
            window_size,
            overlap_ratio[2],
            channel_heads[2],
            spatial_heads[2],
            spatial_dim_head,
            expansion_factor,
            bias,
            layer_norm_type,
        )
        self.down3_4 = Downsample(dim * 4)
        self.latent = _blocks(
            num_blocks[3],
            dim * 8,
            window_size,
            overlap_ratio[3],
            channel_heads[3],
            spatial_heads[3],
            spatial_dim_head,
            expansion_factor,
            bias,
            layer_norm_type,
        )
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = _blocks(
            num_blocks[2],
            dim * 4,
            window_size,
            overlap_ratio[2],
            channel_heads[2],
            spatial_heads[2],
            spatial_dim_head,
            expansion_factor,
            bias,
            layer_norm_type,
        )
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = _blocks(
            num_blocks[1],
            dim * 2,
            window_size,
            overlap_ratio[1],
            channel_heads[1],
            spatial_heads[1],
            spatial_dim_head,
            expansion_factor,
            bias,
            layer_norm_type,
        )
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = _blocks(
            num_blocks[0],
            dim * 2,
            window_size,
            overlap_ratio[0],
            channel_heads[0],
            spatial_heads[0],
            spatial_dim_head,
            expansion_factor,
            bias,
            layer_norm_type,
        )
        self.refinement = _blocks(
            num_refinement_blocks,
            dim * 2,
            window_size,
            overlap_ratio[0],
            channel_heads[0],
            spatial_heads[0],
            spatial_dim_head,
            expansion_factor,
            bias,
            layer_norm_type,
        )
        self.output = nn.Conv2d(dim * 2, output_channels, 3, padding=1, bias=bias)

    @staticmethod
    def _condition(value, parameters):
        if parameters is None:
            return value
        scale, shift = parameters
        return value * (1 + scale) + shift

    def forward(self, value, conditioning=None):
        if conditioning is None:
            conditioning = (None,) * 4

        first = self.patch_embed(value)
        first = self.encoder_level1(self._condition(first, conditioning[0]))
        second = self.down1_2(first)
        second = self.encoder_level2(self._condition(second, conditioning[1]))
        third = self.down2_3(second)
        third = self.encoder_level3(self._condition(third, conditioning[2]))
        latent = self.down3_4(third)
        latent = self.latent(self._condition(latent, conditioning[3]))

        output = self.up4_3(latent)
        output = self.reduce_chan_level3(torch.cat((output, third), dim=1))
        output = self.decoder_level3(output)
        output = self.up3_2(output)
        output = self.reduce_chan_level2(torch.cat((output, second), dim=1))
        output = self.decoder_level2(output)
        output = self.up2_1(output)
        output = self.decoder_level1(torch.cat((output, first), dim=1))
        output = self.refinement(output)
        return self.output(output) + value


class LowRankFourierPSFCode(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        self.code_dim = int(code_dim)
        if self.code_dim <= 0 or self.code_dim % 2:
            raise ValueError("operator code dimension must be a positive even number")

    def forward(self, psf):
        if not isinstance(psf, Tensor) or psf.ndim != 4:
            raise ValueError("psf must be an NCHW tensor")
        if not psf.is_floating_point():
            raise TypeError("psf must be a floating-point tensor")

        psf = psf.mean(dim=1)
        psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        spectrum = torch.fft.rfft2(psf)
        height, frequency_width = spectrum.shape[-2:]
        frequency_count = self.code_dim // 2

        frequencies = []
        for vertical in range(height):
            wrapped_vertical = min(vertical, height - vertical)
            for horizontal in range(frequency_width):
                if vertical == 0 and horizontal == 0:
                    continue
                radius = wrapped_vertical**2 + horizontal**2
                frequencies.append((radius, wrapped_vertical, horizontal, vertical))
        frequencies.sort()
        if len(frequencies) < frequency_count:
            raise ValueError("PSF is too small for the requested operator code")

        coefficients = [
            spectrum[:, vertical, horizontal]
            for _, _, horizontal, vertical in frequencies[:frequency_count]
        ]
        coefficients = torch.stack(coefficients, dim=1)
        code = torch.stack((coefficients.real, coefficients.imag), dim=-1).flatten(1)
        return F.layer_norm(code, (self.code_dim,))


class OperatorFiLM(nn.Module):
    def __init__(self, code_dim, hidden_dim, feature_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(code_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.projections = nn.ModuleList(
            nn.Linear(int(hidden_dim), 2 * int(feature_dim))
            for feature_dim in feature_dims
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, code):
        hidden = self.encoder(code)
        conditioning = []
        for projection in self.projections:
            scale, shift = projection(hidden).chunk(2, dim=1)
            conditioning.append((scale[:, :, None, None], shift[:, :, None, None]))
        return tuple(conditioning)


class PSFFreeXRestormer(nn.Module):
    def __init__(
        self,
        channels=3,
        dim=48,
        num_blocks=(2, 4, 4, 4),
        num_refinement_blocks=4,
        channel_heads=(1, 2, 4, 8),
        spatial_heads=(1, 2, 4, 8),
        overlap_ratio=(0.5, 0.5, 0.5, 0.5),
        window_size=8,
        spatial_dim_head=16,
        expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
        padding_size=64,
        checkpoint_path: str | Path | None = None,
        strict_checkpoint=True,
        output_crop=None,
        operator_prompt="none",
        operator_code_dim=16,
        operator_hidden_dim=128,
    ):
        super().__init__()
        for name, values in (
            ("num_blocks", num_blocks),
            ("channel_heads", channel_heads),
            ("spatial_heads", spatial_heads),
            ("overlap_ratio", overlap_ratio),
        ):
            if len(values) != 4:
                raise ValueError(f"{name} must contain four values")
        if channels <= 0 or dim <= 0 or num_refinement_blocks <= 0:
            raise ValueError("channels, dim and num_refinement_blocks must be positive")
        if padding_size <= 0 or padding_size % (window_size * 8) != 0:
            raise ValueError("padding_size must be a multiple of window_size * 8")
        if output_crop is not None:
            if len(output_crop) != 4:
                raise ValueError("output_crop must contain [top, left, height, width]")
            output_crop = tuple(int(value) for value in output_crop)
            if (
                output_crop[0] < 0
                or output_crop[1] < 0
                or output_crop[2] <= 0
                or output_crop[3] <= 0
            ):
                raise ValueError(
                    "output_crop top/left must be non-negative and "
                    "height/width must be positive"
                )

        self.channels = int(channels)
        self.padding_size = int(padding_size)
        self.output_crop = output_crop
        self.operator_prompt = str(operator_prompt)
        if self.operator_prompt not in {"none", "fourier", "constant"}:
            raise ValueError("operator_prompt must be none, fourier or constant")
        self.network = XRestormer(
            input_channels=self.channels,
            output_channels=self.channels,
            dim=int(dim),
            num_blocks=tuple(int(value) for value in num_blocks),
            num_refinement_blocks=int(num_refinement_blocks),
            channel_heads=tuple(int(value) for value in channel_heads),
            spatial_heads=tuple(int(value) for value in spatial_heads),
            overlap_ratio=tuple(float(value) for value in overlap_ratio),
            window_size=int(window_size),
            spatial_dim_head=int(spatial_dim_head),
            expansion_factor=float(expansion_factor),
            bias=bool(bias),
            layer_norm_type=str(layer_norm_type),
        )
        self.operator_code = None
        self.operator_conditioner = None
        if self.operator_prompt != "none":
            self.operator_code = LowRankFourierPSFCode(operator_code_dim)
            self.operator_conditioner = OperatorFiLM(
                code_dim=operator_code_dim,
                hidden_dim=operator_hidden_dim,
                feature_dims=(int(dim), int(dim) * 2, int(dim) * 4, int(dim) * 8),
            )

        if checkpoint_path is not None:
            self.load_official_checkpoint(
                checkpoint_path,
                strict=strict_checkpoint,
            )

    @property
    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def load_official_checkpoint(
        self,
        checkpoint_path,
        strict=True,
    ):
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Local checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("params"), dict
        ):
            raise ValueError("Official X-Restormer checkpoint must contain params")
        return self.network.load_state_dict(checkpoint["params"], strict=strict)

    def _validate_measurement(self, measurement):
        if not isinstance(measurement, Tensor):
            raise TypeError("measurement must be a torch.Tensor")
        if measurement.ndim != 4:
            raise ValueError("measurement must have NCHW shape")
        if measurement.shape[1] != self.channels:
            raise ValueError(
                f"measurement must have {self.channels} channels, got "
                f"{measurement.shape[1]}"
            )
        if not measurement.is_floating_point():
            raise TypeError("measurement must have a floating-point dtype")
        if not torch.isfinite(measurement).all():
            raise ValueError("measurement must contain only finite values")
        if measurement.numel() and (
            measurement.detach().amin().item() < 0
            or measurement.detach().amax().item() > 1
        ):
            raise ValueError("measurement values must be in [0, 1]")

    def _operator_conditioning(self, measurement, psf):
        if self.operator_prompt == "none":
            return None
        if self.operator_prompt == "fourier":
            if psf is None:
                raise ValueError("fourier operator prompt needs psf in the batch")
            if psf.shape[0] != measurement.shape[0]:
                raise ValueError("measurement and psf batch sizes must match")
            code = self.operator_code(psf.to(dtype=measurement.dtype))
        else:
            code = measurement.new_zeros(
                (measurement.shape[0], self.operator_code.code_dim)
            )
        return self.operator_conditioner(code)

    def forward(self, measurement, psf=None, **batch):
        del batch
        self._validate_measurement(measurement)
        conditioning = self._operator_conditioning(measurement, psf)
        height, width = measurement.shape[-2:]
        pad_height = (-height) % self.padding_size
        pad_width = (-width) % self.padding_size
        padded = F.pad(
            measurement,
            (0, pad_width, 0, pad_height),
            mode="reflect",
        )
        prediction = self.network(padded, conditioning=conditioning)[
            ..., :height, :width
        ].clamp_min(0)
        prediction_max = prediction.amax(dim=(1, 2, 3), keepdim=True)
        prediction = torch.where(
            prediction_max > 0,
            prediction / prediction_max.clamp_min(1e-12),
            prediction,
        )
        if self.output_crop is not None:
            top, left, crop_height, crop_width = self.output_crop
            if top + crop_height > height or left + crop_width > width:
                raise ValueError("output_crop exceeds the reconstructed image bounds")
            prediction = prediction[
                ...,
                top : top + crop_height,
                left : left + crop_width,
            ]
        return {"prediction": prediction}


__all__ = ["PSFFreeXRestormer"]

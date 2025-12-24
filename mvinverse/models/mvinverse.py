import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .dinov2.layers import Mlp
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.dpt_head import DPTHead
from .layers.dpt_head import DPTHeadRes
from .layers.dpt_head import _make_pretrained_resnext101_wsl
from .dinov2.hub.backbones import dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin


class MVInverse(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        pos_type='rope100',
        dec_embed_dim=1024,
        dec_depth=36,
        dec_num_heads=16,
        mlp_ratio=4,
        num_register_tokens=5,
    ):
        super().__init__()
        self.patch_size = 14
        self.dec_embed_dim = dec_embed_dim
        self.num_register_tokens = num_register_tokens
        self.patch_start_idx = num_register_tokens

        self._init_encoder()
        self._init_pos_encoding(pos_type)
        self._init_decoder(dec_embed_dim, dec_depth, dec_num_heads, mlp_ratio)
        self._init_register_tokens(num_register_tokens, dec_embed_dim)
        self._init_heads(dec_embed_dim)
        self._register_normalization_buffers()

    def _init_encoder(self):
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token
        self.res_encoder = _make_pretrained_resnext101_wsl(use_pretrained=False)

    def _init_pos_encoding(self, pos_type: str):
        self.pos_type = pos_type
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("cuRoPE2D not found. Please install it.")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError(f"pos_type '{pos_type}' is not supported.")

    def _init_decoder(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float):
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(depth)])

    def _init_register_tokens(self, num_tokens: int, embed_dim: int):
        self.register_token = nn.Parameter(torch.randn(1, 1, num_tokens, embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

    def _init_heads(self, dec_embed_dim: int):
        dim_in = 2 * dec_embed_dim
        self.albedo_head = DPTHeadRes(dim_in=dim_in, output_dim=3, activation="sigmoid")
        self.metallic_head = DPTHead(dim_in=dim_in, output_dim=1, activation="sigmoid")
        self.roughness_head = DPTHead(dim_in=dim_in, output_dim=1, activation="sigmoid")
        self.normal_head = DPTHead(dim_in=dim_in, output_dim=3, activation="tanh")
        self.shading_head = DPTHeadRes(dim_in=dim_in, output_dim=3, activation="sigmoid")

    def _register_normalization_buffers(self):
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def _get_resnext_features(self, x: torch.Tensor) :
        H, W = x.shape[-2:]
        new_H, new_W = H // 7 * 8, W // 7 * 8
        x_resized = F.interpolate(x, (new_H, new_W), mode='bilinear', align_corners=False)

        layer_1 = self.res_encoder.layer1(x_resized)
        layer_2 = self.res_encoder.layer2(layer_1)
        layer_3 = self.res_encoder.layer3(layer_2)
        layer_4 = self.res_encoder.layer4(layer_3)
        return [layer_1, layer_2, layer_3, layer_4]

    def decode(self, hidden: torch.Tensor, N: int, H: int, W: int):
        BN, hw_patch, C = hidden.shape
        B = BN // N

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(BN, self.num_register_tokens, C)
        hidden = torch.cat([register_token, hidden], dim=1)
        hw_total = hidden.shape[1]

        pos = self.position_getter(BN, H // self.patch_size, W // self.patch_size, hidden.device)
        pos_special = torch.zeros(BN, self.patch_start_idx, 2, device=hidden.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos + 1], dim=1)

        intermediates = []
        hidden_even = None
        for i, blk in enumerate(self.decoder):
            is_even_block = i % 2 == 0
            
            if is_even_block:
                # Intra-view processing
                shape = (BN, hw_total, -1)
            else:
                # Inter-view processing
                shape = (B, N * hw_total, -1)
            
            pos = pos.reshape(*shape)
            hidden = hidden.reshape(*shape)
            hidden = blk(hidden, xpos=pos)

            if is_even_block:
                hidden_even = hidden.reshape(B, N, hw_total, -1)
            else:
                # Concatenate features from pairs of blocks
                hidden_odd = hidden.reshape(B, N, hw_total, -1)
                intermediates.append(torch.cat([hidden_even, hidden_odd], dim=-1))
        
        return intermediates

    def forward(self, imgs: torch.Tensor):
        B, N, _, H, W = imgs.shape
        imgs_flat = imgs.reshape(B * N, 3, H, W)

        imgs_norm = (imgs_flat - self.image_mean) / self.image_std

        # extract features
        hidden_patches = self.encoder(imgs_norm, is_training=True)["x_norm_patchtokens"]
        res_features = self._get_resnext_features(imgs_norm)

        # alternating attention
        intermediates = self.decode(hidden_patches, N, H, W)

        # prediction
        albedo = self.albedo_head(intermediates, imgs, res_features=res_features, patch_start_idx=self.patch_start_idx)
        roughness = self.roughness_head(intermediates, imgs, self.patch_start_idx)
        metallic = self.metallic_head(intermediates, imgs, self.patch_start_idx)
        normal = self.normal_head(intermediates, imgs, self.patch_start_idx)
        shading = self.shading_head(intermediates, imgs, res_features=res_features, patch_start_idx=self.patch_start_idx)

        normal_normalized = F.normalize(normal, p=2, dim=-1, eps=1e-8)

        return {
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "normal": normal_normalized,
            "shading": shading,
        }
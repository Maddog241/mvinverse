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
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 36
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
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
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #    resnext encoder
        # ----------------------
        self.res_encoder = _make_pretrained_resnext101_wsl(use_pretrained=False)

        # ----------------------
        #    Albedo Head
        # ----------------------
        self.albedo_head = DPTHeadRes(
            dim_in=2 * self.dec_embed_dim, 
            output_dim=3, 
            activation="sigmoid", 
        )

                # ----------------------
        #     Metallic Head
        # ----------------------
        self.metallic_head = DPTHead(
            dim_in=2 * self.dec_embed_dim, 
            output_dim=1, 
            activation="sigmoid", 
        ) 

        # ----------------------
        #     Roughness Head
        # ----------------------
        self.roughness_head = DPTHead(
            dim_in=2 * self.dec_embed_dim, 
            output_dim=1, 
            activation="sigmoid", 
        ) 

        # ----------------------
        #     Normal Head
        # ----------------------
        self.normal_head = DPTHead(
            dim_in=2 * self.dec_embed_dim, 
            output_dim=3, 
            activation="tanh", 
        ) 

        # ----------------------
        #     Normal Head
        # ----------------------
        self.shading_head = DPTHeadRes(
            dim_in=2 * self.dec_embed_dim, 
            output_dim=3, 
            activation="sigmoid", 
        )

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        intermediates = []

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            # reshape hidden
            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)
            if i % 2 == 0:
                inter = [hidden.reshape(B, N, hw, -1)]
            else:
                inter.append(hidden.reshape(B, N, hw, -1))
                intermediates.append(torch.concat(inter, dim=-1))

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return intermediates
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        
        # encode by dinov2
        imgs_flat = imgs.reshape(B*N, _, H, W)

        hidden = self.encoder(imgs_flat, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]  # (B*N, S, C) (num_images, num_tokens, channel)
        
        # resize images & caculate multi-res features
        new_H, new_W = H // 7 * 8, W // 7 * 8
        imgs_resize = F.interpolate(imgs_flat, (new_H, new_W))

        layer_1 = self.res_encoder.layer1(imgs_resize)  # [B*N, 256, 2H/7, 2W/7]
        layer_2 = self.res_encoder.layer2(layer_1) # [B*N, 512, H/7, H/7]
        layer_3 = self.res_encoder.layer3(layer_2) # [B*N, 1024, H/14, H/14]
        layer_4 = self.res_encoder.layer4(layer_3) # [B*N, 2048, H/28, W/28]

        res_features = [layer_1, layer_2, layer_3, layer_4]

        intermediates = self.decode(hidden, N, H, W) # (B*N, S + num_reg, 2C)

        albedo = self.albedo_head(intermediates, imgs, res_features=res_features, patch_start_idx=self.patch_start_idx)
        roughness = self.roughness_head(intermediates, imgs.reshape(B, N, _, H, W), self.patch_start_idx) # (B, N, H, W, C)
        metallic = self.metallic_head(intermediates, imgs.reshape(B, N, _, H, W), self.patch_start_idx) # (B, N, H, W, C)
        normal = self.normal_head(intermediates, imgs.reshape(B, N, _, H, W), self.patch_start_idx) # (B, N, H, W, C)
        shading = self.shading_head(intermediates, imgs, res_features=res_features, patch_start_idx=self.patch_start_idx)

        # normalize normals
        normal = F.normalize(normal, p=2, dim=-1, eps=1e-8)

        return dict(
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            normal=normal,
            shading=shading,
        )
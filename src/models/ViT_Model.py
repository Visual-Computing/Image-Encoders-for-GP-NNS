from collections import OrderedDict
import torch
from torch import nn
from torchvision import transforms

"""
Code is based on the OpenAI CLIP repository: https://github.com/openai/CLIP/blob/main/clip/model.py
"""

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        #self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

 
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_checkpointing: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.use_checkpointing = use_checkpointing
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, use_checkpointing: bool = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_checkpointing=use_checkpointing)

        self.ln_post = nn.LayerNorm(width)
        self.pre_embed_dim = width
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class EmbeddingModel(torch.nn.Module):
    
    def __init__(self, backbone, use_proj=True):
        super().__init__()
        self.backbone = backbone
        self.emb_dim = backbone.output_dim
        if not use_proj:
            self.backbone.proj = None
            self.emb_dim = backbone.pre_embed_dim

    
    def forward(self, x):
        
        fv = self.backbone(x)
        return fv

def create_model(model_name="ViT-L", check_pth=None):

    if model_name == "ViT-L":
        vit_model = VisionTransformer(
                    input_resolution=336 ,
                    patch_size=14 ,
                    width=1024 ,
                    layers=24 ,
                    heads=16 ,
                    output_dim=768
                )
        image_size = 336

    elif model_name == "ViT-B":
        vit_model = VisionTransformer(
                    input_resolution=224,
                    patch_size=16,
                    width=768,
                    layers=12,
                    heads=12,
                    output_dim=512
                    )
        image_size = 224

    model = EmbeddingModel(vit_model, False)
    if check_pth is not None:
        model.load_state_dict(torch.load(check_pth))

    model.half()
    model.backbone.half()
    model.eval()

    
    resize_to = int(image_size * 1.125)
    ppc_fn = transforms.Compose([
                transforms.Resize((resize_to, resize_to),  transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
                transforms.ConvertImageDtype(torch.float16)
            ])

    return model, ppc_fn

     
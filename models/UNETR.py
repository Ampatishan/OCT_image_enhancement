import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_size))

    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x + self.pos_embed


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, depth, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.layers = nn.ModuleList([encoder_layer for _ in range(depth)])

    def forward(self, x):
        hidden_states = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            hidden_states.append(x)
        return hidden_states


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
    

class UNETR(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size,
                 emb_size, depth, num_heads, mlp_dim, ext_layers):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer = TransformerEncoder(emb_size, depth, num_heads, mlp_dim)
        self.patch_size = patch_size
        self.ext_layers = ext_layers
        self.embed_dim = emb_size
        self.img_size = img_size
        
        self.decoder0 = nn.Sequential(
            ConvBlock2D(in_channels, 32),
            ConvBlock2D(32, 64)
        )

        self.decoder3 = nn.Sequential(
            DeconvBlock2D(emb_size, 512),
            DeconvBlock2D(512, 256),
            DeconvBlock2D(256, 128)
        )

        self.decoder6 = nn.Sequential(
            DeconvBlock2D(emb_size, 512),
            DeconvBlock2D(512, 256)
        )

        self.decoder9 = DeconvBlock2D(emb_size, 512)

        self.decoder12_upsampler = DeconvBlock2D(emb_size, 512)

        self.decoder9_upsampler = nn.Sequential(
            ConvBlock2D(1024, 512),
            ConvBlock2D(512, 512),
            DeconvBlock2D(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            ConvBlock2D(512, 256),
            DeconvBlock2D(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            ConvBlock2D(256, 128),
            DeconvBlock2D(128, 64)
        )

        self.decoder0_header = nn.Sequential(
            ConvBlock2D(128, 64),
            ConvBlock2D(64, 64),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )



    def forward(self, x):
        x = x.transpose(0,1)
        B = x.shape[0]
        # H = W = x.shape[2]
        patch_tokens = self.patch_embed(x)
        hidden_states = self.transformer(patch_tokens)

        z0 = x
        z3, z6, z9, z12 = [hidden_states[i] for i in self.ext_layers]

        patch_h = patch_w = self.img_size // self.patch_size  # e.g. 14
        z3 = z3.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)
        z6 = z6.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)
        z9 = z9.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)
        z12 = z12.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))

        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))

        z0 = self.decoder0(z0)
        out = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return out

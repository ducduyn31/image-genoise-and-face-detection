from torch import nn

from models.basic_swin import BasicSwinLayer
from models.patch_embed import PatchEmbed
from models.patch_unembed import PatchUnembed


class ResidualSwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(ResidualSwinTransformerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicSwinLayer(dim=dim,
                                             input_resolution=input_resolution,
                                             depth=depth,
                                             num_heads=num_heads,
                                             window_size=window_size,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop, attn_drop=attn_drop,
                                             drop_path=drop_path,
                                             norm_layer=norm_layer,
                                             downsample=downsample,
                                             use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None
        )

        self.patch_unembed = PatchUnembed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None
        )

    def forward(self, x, x_size):
        return self.patch_embed(
            self.conv(
                self.patch_unembed(
                    self.residual_group(x, x_size), x_size
                )
            )
        ) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += self.dim * self.dim * H * W * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops

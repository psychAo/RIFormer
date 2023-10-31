import torch.nn as nn
import torch

class RotateRelEbd(nn.Module):
    def __init__(self, dim, n_circle=2):
        super().__init__()
        diff_add_values = [-1, 0, 1, 2]
        for i in range(n_circle):  # circle: from outside to inside
            circle_mat = nn.Parameter(torch.zeros(1, dim)+diff_add_values[i])
            setattr(self, f"circle_mat{i}", circle_mat)

    def forward(self, x):  
        H, W = x.shape[2], x.shape[3]
        out = torch.zeros_like(x, dtype=torch.float32)
        for i in range(H):
            for j in range(W):
                dis_2_edge = min(i, j, H-i-1, W-j-1)
                cir = getattr(self, f"circle_mat{dis_2_edge}")
                out[:, :, i, j] = x[:, :, i, j].clone() + cir
        return out

def group_1x1_conv(n_group, in_chan, out_chan):
    return nn.Conv2d(
        in_channels=in_chan,
        out_channels=out_chan,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=n_group
    )

class Group_Pixel_Embed(nn.Module):
    def __init__(self, n_group, in_chan, out_chan, spatial_reduce):
        super().__init__()
        self.g1conv = group_1x1_conv(n_group, in_chan, out_chan)
        self.pool = nn.AvgPool2d(kernel_size=spatial_reduce, stride=spatial_reduce)
        self.bn = nn.BatchNorm2d(num_features=out_chan)
        self.ac = nn.LeakyReLU(inplace=True)  

    def forward(self, x):
        x = self.g1conv(x)
        x = self.pool(x)
        x = self.ac(self.bn(x))
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=1, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class RIFormer(nn.Module):
    def __init__(self, emb_type=None, C1=32, C2=64, C3=128, C4=256, mlp_ratio=2, n_class=10):
        super().__init__()
        self.emb_type = emb_type
        self.gpe_1 = Group_Pixel_Embed(n_group=1, in_chan=3, out_chan=C1, spatial_reduce=4)
        # rot_ebd or pos_ebd
        self.rot_ebd_1 = RotateRelEbd(C1, n_circle=4)  # 4,2,1,1
        self.pos_ebd_1 = nn.Parameter(torch.zeros(1, 8*8, C1))  # 8 * 8 = fmap size
        self.block_1_1 = Block(dim=C1, num_heads=1, mlp_ratio=mlp_ratio)
        self.block_1_2 = Block(dim=C1, num_heads=1, mlp_ratio=mlp_ratio)
        self.many_1 = nn.Sequential(self.block_1_1, self.block_1_2)

        self.gpe_2 = Group_Pixel_Embed(n_group=1, in_chan=C1, out_chan=C2, spatial_reduce=2)
        self.rot_ebd_2 = RotateRelEbd(C2, n_circle=2)  # 4,2,1,1
        self.pos_ebd_2 = nn.Parameter(torch.zeros(1, 4*4, C2))
        self.block_2_1 = Block(dim=C2, num_heads=2, mlp_ratio=mlp_ratio)
        self.block_2_2 = Block(dim=C2, num_heads=2, mlp_ratio=mlp_ratio)
        self.many_2 = nn.Sequential(self.block_2_1, self.block_2_2)

        self.gpe_3 = Group_Pixel_Embed(n_group=1, in_chan=C2, out_chan=C3, spatial_reduce=2)
        self.rot_ebd_3 = RotateRelEbd(C3, n_circle=1)  # 4,2,1,1
        self.pos_ebd_3 = nn.Parameter(torch.zeros(1, 2*2, C3))
        self.block_3_1 = Block(dim=C3, num_heads=4, mlp_ratio=mlp_ratio)
        self.block_3_2 = Block(dim=C3, num_heads=4, mlp_ratio=mlp_ratio)
        self.many_3 = nn.Sequential(self.block_3_1, self.block_3_2)

        self.gpe_4 = Group_Pixel_Embed(n_group=1, in_chan=C3, out_chan=C4, spatial_reduce=2)
        self.rot_ebd_4 = RotateRelEbd(C4, n_circle=1)  # 4,2,1,1
        self.pos_ebd_4 = nn.Parameter(torch.zeros(1, 1*1, C4))
        self.block_4_1 = Block(dim=C4, num_heads=8, mlp_ratio=mlp_ratio)
        self.block_4_2 = Block(dim=C4, num_heads=8, mlp_ratio=mlp_ratio)
        self.many_4 = nn.Sequential(self.block_4_1, self.block_4_2)

        self.cls_layer = nn.Linear(C4, n_class)

    def forward(self, x):
        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        
        x = self.gpe_1(x)
        if self.emb_type == 'rot':
            x = self.rot_ebd_1(x)
        x = x.flatten(2).transpose(1, 2)
        if self.emb_type == 'pos':
            x = x + self.pos_ebd_1
        x = self.many_1(x)
        x = x.reshape(B, H//4, W//4, -1).permute(0, 3, 1, 2).contiguous()

        x = self.gpe_2(x)
        if self.emb_type == 'rot':
            x = self.rot_ebd_2(x)
        x = x.flatten(2).transpose(1, 2)
        if self.emb_type == 'pos':
            x = x + self.pos_ebd_2
        x = self.many_2(x)
        x = x.reshape(B, H//8, W//8, -1).permute(0, 3, 1, 2).contiguous()

        x = self.gpe_3(x)
        if self.emb_type == 'rot':
            x = self.rot_ebd_3(x)
        x = x.flatten(2).transpose(1, 2)
        if self.emb_type == 'pos':
            x = x + self.pos_ebd_3
        x = self.many_3(x)
        x = x.reshape(B, H//16, W//16, -1).permute(0, 3, 1, 2).contiguous()

        x = self.gpe_4(x)
        if self.emb_type == 'rot':
            x = self.rot_ebd_4(x)
        x = x.flatten(2).transpose(1, 2)
        if self.emb_type == 'pos':
            x = x + self.pos_ebd_4
        x = self.many_4(x)
        x = x.reshape(B, H//32, W//32, -1).permute(0, 3, 1, 2).contiguous()
        x = x.flatten(start_dim=1)

        x = self.cls_layer(x)

        return x

if __name__ == "__main__":
    input_t = torch.rand(8, 3, 32, 32)
    net = RIFormer(emb_type='rot', n_class=10)
    print(net(input_t).shape)

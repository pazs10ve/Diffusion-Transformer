import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float):
    return torch.linspace(beta_start, beta_end, timesteps)


class Patchify(nn.Module):
    def __init__(self, patch_size : int):
        super(Patchify, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image Dimensions must be divisible by patch size"
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, num_patches, -1)
    

"""
patch_size = 8
patches = Patchify(patch_size=patch_size)
x = torch.randn(1, 32, 32, 4)
print(patches(x).shape) --> torch.Size([1, 16, 256])
"""


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim : int, num_heads : int, mlp_ratio : float = 4.0) -> None:
        super(DiTBlock, self).__init__()
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = num_heads, batch_first = True)
        self.ffn = nn.Sequential(
            nn.Linear(in_features = hidden_dim, out_features = int(hidden_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(in_features = int(hidden_dim * mlp_ratio), out_features = hidden_dim)
        )
        self.conditioning_mlp = nn.Sequential(
            nn.Linear(in_features = hidden_dim, out_features = 6 * hidden_dim),
            nn.Linear(in_features = 6 * hidden_dim, out_features = 6 * hidden_dim)
        )


    def forward(self, x, cond):
        gammas1, alphas1, betas1, gammas2, alphas2, betas2 = self.conditioning_mlp(cond).chunk(6, dim = -1)
        norm_x = self.layer_norm1(x)
        norm_x = norm_x * gammas1 + betas1
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        attn_out = attn_out * alphas1
        norm_x = norm_x + attn_out

        x = self.layer_norm2(norm_x)
        x = x * gammas2 + betas2
        x = self.ffn(x)
        x = x * alphas2
        x = x + norm_x

        return x 


"""
num_heads = 6
conditioning_dim = 384
dit_block = DiTBlock(hidden_num, num_heads, conditioning_dim)
x = torch.randn(1, 384)
cond = torch.randn(1, 384)
print(dit_block(x, cond).shape)
"""


class DiT(nn.Module):
    def __init__(self, input_dim : int, label_dim : int, patch_size : int, cond_dim : int, hidden_dim : int, num_heads : int, num_layers : int) -> None:
        super(DiT, self).__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.label_proj = nn.Linear(in_features = label_dim, out_features = cond_dim)
        self.patchify = Patchify(patch_size = patch_size)
        patch_dim = 4 * patch_size * patch_size
        self.linear_proj = nn.Linear(in_features = patch_dim, out_features = hidden_dim)
        self.cond_proj = nn.Linear(in_features = cond_dim, out_features = hidden_dim)
        self.mha_blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.out_proj1 = nn.Linear(in_features = hidden_dim, out_features = patch_dim)
        self.out_proj2 = nn.Linear(in_features = hidden_dim, out_features = patch_dim)
        self.betas = linear_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=2e-2)


    def forward(self, x, y, timestep):
        cond = self.label_proj(y)
        #print(x.shape)
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        x = self.patchify(x)
        x = self.linear_proj(x)
        cond = self.cond_proj(cond)
        t_emb = self.get_timestep_embedding(timestep, x.shape[2])
        cond_t = cond.unsqueeze(1) + t_emb.unsqueeze(1)
        x = x + cond_t

        for block in self.mha_blocks:
            x = block(x, cond_t)

        x = self.layer_norm(x)
        x1 = self.out_proj1(x)
        x2 = self.out_proj2(x)

        B, T, _ = x1.shape
        x1 = x1.view(B, int(math.sqrt(T)), int(math.sqrt(T)), self.patch_size, self.patch_size, 4)
        x1 = x1.permute(0, 5, 1, 3, 2, 4).contiguous()
        x1 = x1.view(B, self.input_dim, self.input_dim, 4)

        B, T, _ = x2.shape
        x2 = x2.view(B, int(math.sqrt(T)), int(math.sqrt(T)), self.patch_size, self.patch_size, 4)
        x2 = x2.permute(0, 5, 1, 3, 2, 4).contiguous()
        x2 = x2.view(B, self.input_dim, self.input_dim, 4)

        return x1, x2


    @staticmethod
    def get_timestep_embedding(timestep, hidden_dim):
        half_dim = hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = timestep.device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim = -1)
        return emb
    
    def forward_diffusion(self, x0, t):
        beta_t = self.betas[t]  
        noise = torch.randn_like(x0)  
        #print(f"x0 shape: {x0.shape}")
        #print(f"noise shape: {noise.shape}")
        #print(f"beta_t shape: {beta_t.shape}")
        beta_t = beta_t.view(-1, 1, 1, 1)
        x_t = torch.sqrt(1 - beta_t) * x0 + torch.sqrt(beta_t) * noise
        return x_t, noise



def get_model(config : dict):
    batch_size = config['batch_size']
    input_dim = config['input_dim']
    label_dim = config['label_dim']
    channels = config['channels']
    patch_size = config['patch_size']
    cond_dim = config['cond_dim']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    num_layers = config['num_layers']

    model = DiT(input_dim, label_dim, patch_size, cond_dim, hidden_dim, num_heads, num_layers)
    return model

"""
batch_size = 256
input_dim = 32
label_dim = 1000
channels = 4
patch_size = 8
cond_dim = 384
hidden_dim = 384
num_heads = 6
num_layers = 12

model = DiT(input_dim, label_dim, patch_size, cond_dim, hidden_dim, num_heads, num_layers)
x =  torch.randn(batch_size, input_dim, input_dim, channels)
y = torch.randn(batch_size, label_dim)
timestep = torch.Tensor([0.5])
noise, covariance = model(x, y, timestep)
print(noise.shape, covariance.shape) --> [256, 32, 32, 4], [256, 32, 32, 4]

"""




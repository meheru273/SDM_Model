import torch
import torch.nn as nn
import torch.nn.functional as F
from conditional_encoding import ConditionEncoder


class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        
    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].reshape(batch_size)
        
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
            
        return sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_cumprod[t] * noise_pred)) / self.sqrt_alpha_cumprod[t]
        x0 = torch.clamp(x0, -1.0, 1.0)
        
        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cumprod[t]))
        mean = mean / torch.sqrt(self.alphas[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cumprod[t - 1]) / (1 - self.alpha_cumprod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0


def get_time_embeddings(time_steps, t_emb_dim):
    factor = 10000 ** (
        torch.arange(start=0, end=t_emb_dim // 2, device=time_steps.device) / (t_emb_dim // 2)
    )
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), dim=-1)
    return t_emb



class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()
        
    def forward(self, x, t_emb):
        out = x
        resnet_input = out
        
        out = self.resnet_conv1(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv2(out)
        out = out + self.residual_input_conv(resnet_input)
        
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        
        # Store skip connection before downsampling
        skip = out
        out = self.down_sample_conv(out)
        
        return out, skip


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])
        self.resnet_conv2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        ])
        
    def forward(self, x, t_emb):
        out = x
        
        # First ResNet block
        resnet_input = out
        out = self.resnet_conv1[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv2[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        # Attention
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        
        # Second ResNet block
        resnet_input = out
        out = self.resnet_conv1[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv2[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
        
        return out



class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads, skip_channels=None):
        super().__init__()
        self.up_sample = up_sample
        self.skip_channels = skip_channels
        
        # Calculate actual input channels (includes skip connection if present)
        actual_in_channels = in_channels + (skip_channels if skip_channels else 0)
        
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(8, actual_in_channels),
            nn.SiLU(),
            nn.Conv2d(actual_in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(actual_in_channels, out_channels, kernel_size=1)
        
    def forward(self, x, t_emb, skip=None):
        # Upsample first
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Then concatenate skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        out = x
        resnet_input = out
        
        out = self.resnet_conv1(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv2(out)
        out = out + self.residual_input_conv(resnet_input)
        
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        
        return out


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, channel_mults=(1, 2, 4, 8),
                 t_emb_dim=128, num_heads=4, gridsat_channels=1, era5_channels=4, img_size=64,
                 condition_embed_dim=128):
        super().__init__()
        
        self.t_emb_dim = t_emb_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Condition encoder for ERA5, GRIDSAT, timestamps, and storm names
        self.condition_encoder = ConditionEncoder(
            gridsat_channels=gridsat_channels,
            era5_channels=era5_channels,
            img_size=img_size,
            embed_dim=condition_embed_dim,
            output_channels=base_channels
        )
        
        # Time embedding layers (for diffusion timestep)
        self.time_embed = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim * 4)
        )
        
        # Combine diffusion time embedding with condition context embedding
        self.combined_embed = nn.Sequential(
            nn.Linear(t_emb_dim * 4 + condition_embed_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim * 4)
        )
        
        # Initial convolution (concatenates encoded condition and noisy target)
        self.init_conv = nn.Conv2d(base_channels + in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoders = nn.ModuleList()
        curr_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels_enc = base_channels * mult
            self.encoders.append(
                DownBlock(curr_channels, out_channels_enc, t_emb_dim * 4, 
                             down_sample=(i < len(channel_mults) - 1), num_heads=num_heads)
            )
            curr_channels = out_channels_enc
        
        # Middle
        self.mid_block = MidBlock(curr_channels, curr_channels, t_emb_dim * 4, num_heads)
        
        # Decoder
        self.decoders = nn.ModuleList()
        reversed_mults = list(reversed(channel_mults))[1:]  # Skip bottleneck level
        
        for decoder_idx, mult in enumerate(reversed_mults):
            out_channels_dec = base_channels * mult
            skip_channels = out_channels_dec  # Skip connection has same channels as output
                
            self.decoders.append(
                UpBlock(curr_channels, out_channels_dec, t_emb_dim * 4,
                           up_sample=True, num_heads=num_heads, skip_channels=skip_channels)
            )
            curr_channels = out_channels_dec
            
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, noisy_target, gridsat, era5, timestamps, storm_names, t):
        """
        Args:
            noisy_target: (B, 1, H, W) - Noisy target image at timestep t
            gridsat: (B, 1, H, W) - Current GRIDSAT satellite imagery
            era5: (B, 4, H, W) - Current ERA5 atmospheric data
            timestamps: List of timestamp strings
            storm_names: List of storm name strings
            t: (B,) - Diffusion timestep
        
        Returns:
            Predicted noise: (B, 1, H, W)
        """
        # Encode conditions (ERA5, GRIDSAT, timestamps, storm names)
        encoded_condition, context_emb = self.condition_encoder(gridsat, era5, timestamps, storm_names)
        
        # Concatenate encoded condition with noisy target
        x = torch.cat([encoded_condition, noisy_target], dim=1)
        
        # Get diffusion time embedding
        t_emb = get_time_embeddings(t, self.t_emb_dim)
        t_emb = self.time_embed(t_emb)
        
        # Combine diffusion time embedding with condition context
        combined_emb = torch.cat([t_emb, context_emb], dim=-1)
        combined_emb = self.combined_embed(combined_emb)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        skips = []
        for i, encoder in enumerate(self.encoders):
            x, skip = encoder(x, combined_emb)
            skips.append(skip)
        
        # Middle
        x = self.mid_block(x, combined_emb)
        
        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skips) - 2 - i  # -2 because we skip the bottleneck level
            skip = skips[skip_idx] if skip_idx >= 0 else None
            x = decoder(x, combined_emb, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

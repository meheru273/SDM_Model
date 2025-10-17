import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F

##########data loader########
class CycloneDataset(Dataset):
    def __init__(self, root_dir, years, gridsat_type='GRIDSAT_data.npy', 
                 era5_type='ERA5_data.npy', transform=None, max_samples=None,
                 test_storm=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.gridsat_type = gridsat_type
        self.era5_type = era5_type
        self.transform = transform
        self.samples = []
        self.test_storm = test_storm
        self.mode = mode
        
        self._load_samples(years, max_samples)
        
    def _load_samples(self, years, max_samples):
        """Load all valid samples from the dataset"""
        for year_folder in years:
            year_path = self.root_dir / year_folder
            if not year_path.exists():
                print(f"Warning: Year folder {year_folder} not found at {year_path}")
                continue
            
            # Handle nested folder structure (e.g., 2005_0/2005_0/)
            nested_path = year_path / year_folder
            if nested_path.exists():
                year_path = nested_path
                
            for cyclone_folder in year_path.iterdir():
                if not cyclone_folder.is_dir():
                    continue
                
                storm_name = cyclone_folder.name
                
                if self.test_storm:
                    if self.mode == 'train' and storm_name == self.test_storm:
                        continue
                    elif self.mode == 'val' and storm_name != self.test_storm:
                        continue
                
                timestep_folders = sorted([f for f in cyclone_folder.iterdir() if f.is_dir()])
                
                
                for i in range(len(timestep_folders) - 1):
                    current_time = timestep_folders[i]
                    next_time = timestep_folders[i + 1]
                    
                    current_gridsat = current_time / self.gridsat_type
                    current_era5 = current_time / self.era5_type
                    next_gridsat = next_time / self.gridsat_type
                    
                    # Check if all files exist
                    if (current_gridsat.exists() and current_era5.exists() and 
                        next_gridsat.exists()):
                        
                        # Extract timestamp from folder name
                        timestamp = current_time.name
                        
                        self.samples.append({
                            'current_gridsat': str(current_gridsat),
                            'current_era5': str(current_era5),
                            'next_gridsat': str(next_gridsat),
                            'timestamp': timestamp,
                            'storm_name': storm_name,
                            'year': year_folder
                        })
                        
                        if max_samples and len(self.samples) >= max_samples:
                            return
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load data and replace NaN with zeros
        current_gridsat = np.load(sample_info['current_gridsat']).astype(np.float32)
        current_gridsat = np.nan_to_num(current_gridsat, nan=0.0)
        
        current_era5 = np.load(sample_info['current_era5']).astype(np.float32)
        current_era5 = np.nan_to_num(current_era5, nan=0.0)
        
        next_gridsat = np.load(sample_info['next_gridsat']).astype(np.float32)
        next_gridsat = np.nan_to_num(next_gridsat, nan=0.0)
        
        # Ensure correct dimensions
        if current_gridsat.ndim == 2:
            current_gridsat = current_gridsat[np.newaxis, ...]  # Add channel dim
        if current_era5.ndim == 2:
            current_era5 = current_era5[np.newaxis, ...]
        if next_gridsat.ndim == 2:
            next_gridsat = next_gridsat[np.newaxis, ...]
        
        # Convert to tensors
        current_gridsat = torch.from_numpy(current_gridsat)
        current_era5 = torch.from_numpy(current_era5)
        next_gridsat = torch.from_numpy(next_gridsat)
        
        # Resize to target size
        target_size = (64, 64)
        
        current_gridsat = F.interpolate(
            current_gridsat.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        current_era5 = F.interpolate(
            current_era5.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        next_gridsat = F.interpolate(
            next_gridsat.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Normalize to [-1, 1]
        eps = 1e-8
        
        # GRIDSAT normalization
        gs_min, gs_max = float(current_gridsat.min()), float(current_gridsat.max())
        if gs_max - gs_min > eps:
            current_gridsat = (current_gridsat - gs_min) / (gs_max - gs_min + eps)
            current_gridsat = current_gridsat * 2 - 1
        
        # ERA5 normalization (channel-wise)
        for c in range(current_era5.shape[0]):
            e_min = float(current_era5[c].min())
            e_max = float(current_era5[c].max())
            if e_max - e_min > eps:
                current_era5[c] = (current_era5[c] - e_min) / (e_max - e_min + eps)
                current_era5[c] = current_era5[c] * 2 - 1
        
        # Target normalization
        ngs_min, ngs_max = float(next_gridsat.min()), float(next_gridsat.max())
        if ngs_max - ngs_min > eps:
            next_gridsat = (next_gridsat - ngs_min) / (ngs_max - ngs_min + eps)
            next_gridsat = next_gridsat * 2 - 1
        
        if self.transform:
            current_gridsat = self.transform(current_gridsat)
            current_era5 = self.transform(current_era5)
            next_gridsat = self.transform(next_gridsat)
        
        return {
            'gridsat': current_gridsat,      # (1, 64, 64)
            'era5': current_era5,            # (4, 64, 64)
            'target': next_gridsat,          # (1, 64, 64)
            'timestamp': sample_info['timestamp'],
            'storm_name': sample_info['storm_name'],
            'year': sample_info['year']
        }



def get_train_val_dataloaders(root_dir, batch_size=4, num_workers=2, 
                               train_years=['2005_0', '2016_0', '2022_0'], 
                               val_years=['2022_0'],
                               gridsat_type='GRIDSAT_data.npy',
                               era5_type='ERA5_data.npy',
                               max_samples=None,
                               test_storm="2022349N13068"):
    
    print(f"Using test storm '{test_storm}' as validation set")
    
    train_dataset = CycloneDataset(
        root_dir=root_dir,
        years=train_years,
        gridsat_type=gridsat_type,
        era5_type=era5_type,
        max_samples=max_samples,
        test_storm=test_storm,
        mode='train'
    )
    
    val_dataset = CycloneDataset(
        root_dir=root_dir,
        years=val_years,
        gridsat_type=gridsat_type,
        era5_type=era5_type,
        test_storm=test_storm,
        mode='val'
    )
    
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
###########evaluation and metrices#########
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg
import matplotlib.pyplot as plt
from pathlib import Path

class MetricsCalculator:
    """Calculate various metrics for image generation quality"""
    
    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error"""
        return torch.mean(torch.abs(pred - target)).item()
    
    @staticmethod
    def mse(pred, target):
        """Mean Squared Error"""
        return F.mse_loss(pred, target).item()
    
    @staticmethod
    def rmse(pred, target):
        """Root Mean Squared Error"""
        return torch.sqrt(F.mse_loss(pred, target)).item()
    
    @staticmethod
    def psnr(pred, target, max_val=2.0):
        """Peak Signal-to-Noise Ratio
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            max_val: Maximum possible value (2.0 for [-1, 1] range)
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def ssim(pred, target, window_size=11, size_average=True):
        """Structural Similarity Index
        Simplified SSIM implementation
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                             for x in range(window_size)])
        gauss = gauss / gauss.sum()
        
        # Create 2D window
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.to(pred.device)
        
        # Expand window for all channels
        channel = pred.size(1)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        
        # Calculate means
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean().item()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class FIDCalculator:
    """Calculate Fréchet Inception Distance"""
    def __init__(self, device='cuda'):
        self.device = device
        # Load pretrained Inception v3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove final layer
        self.inception.eval()
        self.inception.to(device)
    
    def get_features(self, images):
        """Extract features from Inception v3
        Args:
            images: (B, 1, H, W) in [-1, 1] range
        Returns:
            Features: (B, 2048)
        """
        # Convert grayscale to RGB
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 (Inception input size)
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1] then to Inception's expected range
        images = (images + 1) / 2.0  # [-1, 1] -> [0, 1]
        
        with torch.no_grad():
            features = self.inception(images)
        
        return features.cpu().numpy()
    
    def calculate_fid(self, real_features, fake_features):
        """Calculate FID score
        Args:
            real_features: Features from real images (N, 2048)
            fake_features: Features from generated images (N, 2048)
        Returns:
            FID score
        """
        # Calculate mean and covariance
        mu1 = np.mean(real_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        
        mu2 = np.mean(fake_features, axis=0)
        sigma2 = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return fid


def evaluate_model(model, dataloader, scheduler, device, condition_encoder, num_samples=100):
    """Evaluate model on validation set
    Args:
        model: Trained diffusion model
        dataloader: Validation dataloader
        scheduler: Noise scheduler
        device: Device to run on
        condition_encoder: Condition encoder module
        num_samples: Number of samples to evaluate
    Returns:
        Dictionary of metrics
    """
    model.eval()
    condition_encoder.eval()
    
    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'psnr': [],
        'ssim': []
    }
    
    calc = MetricsCalculator()
    all_real = []
    all_fake = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break
            
            gridsat = batch['gridsat'].to(device)
            era5 = batch['era5'].to(device)
            target = batch['target'].to(device)
            timestamps = batch['timestamp']
            storm_names = batch['storm_name']
            
            # Encode conditions
            encoded_cond, context_emb = condition_encoder(
                gridsat, era5, timestamps, storm_names
            )
            
            # Generate samples using DDPM sampling
            batch_size = target.shape[0]
            generated = torch.randn_like(target).to(device)
            
            # Reverse diffusion process
            for t in reversed(range(scheduler.num_timesteps)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise
                noise_pred = model(generated, encoded_cond, t_batch)
                
                # Sample previous timestep
                generated, _ = scheduler.sample_prev_timestep(generated, noise_pred, t)
            
            # Calculate metrics
            metrics['mae'].append(calc.mae(generated, target))
            metrics['mse'].append(calc.mse(generated, target))
            metrics['rmse'].append(calc.rmse(generated, target))
            metrics['psnr'].append(calc.psnr(generated, target))
            metrics['ssim'].append(calc.ssim(generated, target))
            
            # Store for FID calculation
            all_real.append(target)
            all_fake.append(generated)
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    # Calculate FID if enough samples
    if len(all_real) > 0:
        all_real = torch.cat(all_real, dim=0)
        all_fake = torch.cat(all_fake, dim=0)
        
        try:
            fid_calc = FIDCalculator(device)
            real_features = fid_calc.get_features(all_real)
            fake_features = fid_calc.get_features(all_fake)
            fid_score = fid_calc.calculate_fid(real_features, fake_features)
            avg_metrics['fid'] = fid_score
        except Exception as e:
            print(f"FID calculation failed: {e}")
            avg_metrics['fid'] = None
    
    return avg_metrics


def save_comparison_images(model, dataloader, scheduler, condition_encoder, 
                          device, save_dir, num_samples=8):
    """Generate and save side-by-side comparison images
    Args:
        model: Trained model
        dataloader: Dataloader
        scheduler: Noise scheduler
        condition_encoder: Condition encoder
        device: Device
        save_dir: Directory to save images
        num_samples: Number of samples to save
    """
    model.eval()
    condition_encoder.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        
        gridsat = batch['gridsat'][:num_samples].to(device)
        era5 = batch['era5'][:num_samples].to(device)
        target = batch['target'][:num_samples].to(device)
        timestamps = batch['timestamp'][:num_samples]
        storm_names = batch['storm_name'][:num_samples]
        
        # Encode conditions
        encoded_cond, context_emb = condition_encoder(
            gridsat, era5, timestamps, storm_names
        )
        
        # Generate samples
        batch_size = target.shape[0]
        generated = torch.randn_like(target).to(device)
        
        for t in reversed(range(scheduler.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(generated, encoded_cond, t_batch)
            generated, _ = scheduler.sample_prev_timestep(generated, noise_pred, t)
        
        # Move to CPU and convert to numpy
        gridsat_np = gridsat.cpu().numpy()
        target_np = target.cpu().numpy()
        generated_np = generated.cpu().numpy()
        
        # Create comparison figure
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        
        for i in range(num_samples):
            # Input (GridSat at time t)
            axes[i, 0].imshow(gridsat_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 0].set_title(f'Input t\n{storm_names[i]}\n{timestamps[i]}')
            axes[i, 0].axis('off')
            
            # Target (GridSat at time t+1)
            axes[i, 1].imshow(target_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 1].set_title(f'Ground Truth t+1')
            axes[i, 1].axis('off')
            
            # Generated (Predicted t+1)
            axes[i, 2].imshow(generated_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 2].set_title(f'Predicted t+1')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison images to {save_dir / 'comparison.png'}")


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")
##########condition encoding########
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

class SpatialPositionalEncoding(nn.Module):
    """2D Positional encoding for image patches"""
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # Create 2D positional encoding
        pos_encoding = self._get_2d_positional_encoding(height, width, channels)
        self.register_buffer('pos_encoding', pos_encoding)
    
    def _get_2d_positional_encoding(self, h, w, d_model):
        """Generate 2D sinusoidal positional encoding"""
        pe = torch.zeros(d_model, h, w)
        
        # Create position indices
        y_pos = torch.arange(0, h).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(0, w).unsqueeze(0).repeat(h, 1)
        
        # Calculate division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        
        # Apply sin to even indices in the depth
        pe[0::4, :, :] = torch.sin(y_pos * div_term[0::2].unsqueeze(-1).unsqueeze(-1))
        pe[1::4, :, :] = torch.cos(y_pos * div_term[0::2].unsqueeze(-1).unsqueeze(-1))
        
        # Apply sin to odd indices in the depth  
        pe[2::4, :, :] = torch.sin(x_pos * div_term[1::2].unsqueeze(-1).unsqueeze(-1))
        pe[3::4, :, :] = torch.cos(x_pos * div_term[1::2].unsqueeze(-1).unsqueeze(-1))
        
        return pe
    
    def forward(self, x):
        """Add positional encoding to input
        Args:
            x: (B, C, H, W)
        Returns:
            x with positional encoding added: (B, C, H, W)
        """
        return x + self.pos_encoding.unsqueeze(0)


class TemporalEncoding(nn.Module):
    """Encode timestamp information"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Separate embeddings for different time components
        self.year_embed = nn.Embedding(50, embed_dim // 4)  # 2000-2050
        self.month_embed = nn.Embedding(12, embed_dim // 4)
        self.day_embed = nn.Embedding(31, embed_dim // 4)
        self.hour_embed = nn.Embedding(24, embed_dim // 4)
        
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string to components
        Args:
            timestamp_str: e.g., "2021-11-19 06_00_00"
        Returns:
            year, month, day, hour indices
        """
        # Handle both formats: "2021-11-19 06_00_00" and "2021-11-19_06_00_00"
        timestamp_str = timestamp_str.replace('_', ' ', 1)
        dt = datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%d %H_%M_%S")
        
        return (
            dt.year - 2000,  # Normalize to 0-50 range
            dt.month - 1,     # 0-11
            dt.day - 1,       # 0-30
            dt.hour           # 0-23
        )
    
    def forward(self, timestamps):
        """Encode timestamps
        Args:
            timestamps: List of timestamp strings or tensor of indices (B, 4)
        Returns:
            Temporal embeddings: (B, embed_dim)
        """
        if isinstance(timestamps, (list, tuple)) and isinstance(timestamps[0], str):
            # Parse string timestamps
            batch_size = len(timestamps)
            years, months, days, hours = [], [], [], []
            
            for ts in timestamps:
                y, m, d, h = self.parse_timestamp(ts)
                years.append(y)
                months.append(m)
                days.append(d)
                hours.append(h)
            
            years = torch.tensor(years, dtype=torch.long)
            months = torch.tensor(months, dtype=torch.long)
            days = torch.tensor(days, dtype=torch.long)
            hours = torch.tensor(hours, dtype=torch.long)
        else:
            # Already tensor of indices
            years = timestamps[:, 0].long()
            months = timestamps[:, 1].long()
            days = timestamps[:, 2].long()
            hours = timestamps[:, 3].long()
        
        # Move to same device as embeddings
        device = self.year_embed.weight.device
        years = years.to(device)
        months = months.to(device)
        days = days.to(device)
        hours = hours.to(device)
        
        # Get embeddings
        year_emb = self.year_embed(years)
        month_emb = self.month_embed(months)
        day_emb = self.day_embed(days)
        hour_emb = self.hour_embed(hours)
        
        # Concatenate all components
        temporal_emb = torch.cat([year_emb, month_emb, day_emb, hour_emb], dim=-1)
        
        return self.projection(temporal_emb)


class StormEncoding(nn.Module):
    """Encode storm identity"""
    def __init__(self, max_storms=1000, embed_dim=128):
        super().__init__()
        self.storm_embed = nn.Embedding(max_storms, embed_dim)
        self.storm_name_to_idx = {}
        self.next_idx = 0
    
    def get_storm_idx(self, storm_name):
        """Get or create index for storm name"""
        if storm_name not in self.storm_name_to_idx:
            self.storm_name_to_idx[storm_name] = self.next_idx
            self.next_idx += 1
        return self.storm_name_to_idx[storm_name]
    
    def forward(self, storm_names):
        """Encode storm names
        Args:
            storm_names: List of storm name strings or tensor of indices (B,)
        Returns:
            Storm embeddings: (B, embed_dim)
        """
        if isinstance(storm_names, (list, tuple)) and isinstance(storm_names[0], str):
            # Convert names to indices
            indices = [self.get_storm_idx(name) for name in storm_names]
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            indices = storm_names.long()
        
        # Move to same device as embeddings
        device = self.storm_embed.weight.device
        indices = indices.to(device)
        
        return self.storm_embed(indices)


class ConditionEncoder(nn.Module):
    """Complete condition encoder combining spatial, temporal, and storm information"""
    def __init__(self, 
                 gridsat_channels=1,
                 era5_channels=4,
                 img_size=64,
                 embed_dim=128,
                 output_channels=64):
        super().__init__()
        
        # Spatial positional encoding for images
        self.spatial_pos_gridsat = SpatialPositionalEncoding(
            gridsat_channels, img_size, img_size
        )
        self.spatial_pos_era5 = SpatialPositionalEncoding(
            era5_channels, img_size, img_size
        )
        
        # Temporal and storm encodings
        self.temporal_encoder = TemporalEncoding(embed_dim)
        self.storm_encoder = StormEncoding(embed_dim=embed_dim)
        
        # Combine temporal and storm embeddings
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Process concatenated images with positional encoding
        total_input_channels = gridsat_channels + era5_channels
        self.image_encoder = nn.Sequential(
            nn.Conv2d(total_input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        )
        
        # Project context embedding to spatial features
        self.context_to_spatial = nn.Sequential(
            nn.Linear(embed_dim, output_channels * img_size * img_size),
            nn.SiLU()
        )
        
    def forward(self, gridsat, era5, timestamps, storm_names):
        """
        Args:
            gridsat: (B, 1, H, W) - Satellite IR imagery
            era5: (B, 4, H, W) - ERA5 atmospheric data
            timestamps: List of timestamp strings or tensor (B, 4)
            storm_names: List of storm names or tensor (B,)
        
        Returns:
            encoded_condition: (B, output_channels, H, W)
            context_embedding: (B, embed_dim) - for cross-attention
        """
        batch_size = gridsat.shape[0]
        img_size = gridsat.shape[-1]
        
        # Add spatial positional encoding to images
        gridsat_pos = self.spatial_pos_gridsat(gridsat)
        era5_pos = self.spatial_pos_era5(era5)
        
        # Concatenate and encode images
        combined_images = torch.cat([gridsat_pos, era5_pos], dim=1)
        image_features = self.image_encoder(combined_images)
        
        # Encode temporal and storm information
        temporal_emb = self.temporal_encoder(timestamps)
        storm_emb = self.storm_encoder(storm_names)
        
        # Combine temporal and storm context
        context_emb = torch.cat([temporal_emb, storm_emb], dim=-1)
        context_emb = self.context_projection(context_emb)
        
        # Project context to spatial features and add to image features
        context_spatial = self.context_to_spatial(context_emb)
        context_spatial = context_spatial.view(batch_size, -1, img_size, img_size)
        
        # Combine image features with context
        encoded_condition = image_features + context_spatial
        
        return encoded_condition, context_emb
##########diffusion model and scheduler##########
import torch
import torch.nn as nn
import torch.nn.functional as F


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
######## loss functions#######
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Loss function for diffusion model training"""
    
    def __init__(self, loss_type='mse', use_l1=False, l1_weight=0.1):
        """
        Args:
            loss_type: Type of loss ('mse', 'l1', 'smooth_l1', 'huber')
            use_l1: If True and loss_type='mse', add L1 regularization
            l1_weight: Weight for L1 loss component
        """
        super().__init__()
        self.loss_type = loss_type
        self.use_l1 = use_l1
        self.l1_weight = l1_weight
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss()
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose from ['mse', 'l1', 'smooth_l1', 'huber']")
    
    def forward(self, pred_noise, true_noise):
        """
        Calculate diffusion loss
        
        Args:
            pred_noise: Predicted noise from model (B, C, H, W)
            true_noise: Ground truth noise (B, C, H, W)
        
        Returns:
            Total loss (scalar)
        """
        # Base loss (MSE, L1, etc.)
        loss = self.base_loss(pred_noise, true_noise)
        
        # Optional additional L1 loss for MSE training (helps with sharper predictions)
        if self.use_l1 and self.loss_type == 'mse':
            l1_loss = F.l1_loss(pred_noise, true_noise)
            loss = loss + self.l1_weight * l1_loss
        
        return loss


class PerceptualLoss(nn.Module):
    """
    Simple perceptual loss using feature matching
    Helps preserve structural information in generated images
    """
    
    def __init__(self, feature_layers=[64, 128], in_channels=1):
        """
        Args:
            feature_layers: List of feature dimensions for each layer
            in_channels: Number of input channels
        """
        super().__init__()
        
        self.feature_extractors = nn.ModuleList()
        prev_channels = in_channels
        
        for features in feature_layers:
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            )
            prev_channels = features
    
    def forward(self, pred, target):
        """
        Calculate perceptual loss
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
        
        Returns:
            Perceptual loss (scalar)
        """
        loss = 0
        pred_feat = pred
        target_feat = target
        
        for extractor in self.feature_extractors:
            pred_feat = extractor(pred_feat)
            target_feat = extractor(target_feat)
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(self.feature_extractors)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss
    Measures perceptual similarity between images
    """
    
    def __init__(self, window_size=11, size_average=True):
        """
        Args:
            window_size: Size of Gaussian window
            size_average: If True, average over batch
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _create_window(self, window_size, channel):
        """Create Gaussian window for SSIM calculation"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2)))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred, target):
        """
        Calculate SSIM loss (1 - SSIM)
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
        
        Returns:
            SSIM loss (scalar)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        channel = pred.size(1)
        if self.window.device != pred.device or self.window.dtype != pred.dtype:
            self.window = self._create_window(self.window_size, channel).to(
                device=pred.device, dtype=pred.dtype
            )
        
        # Calculate means
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class CombinedDiffusionLoss(nn.Module):
    """
    Combined loss with multiple components:
    - Diffusion loss (MSE/L1 on noise prediction)
    - Optional perceptual loss
    - Optional SSIM loss
    """
    
    def __init__(self, 
                 loss_type='mse',
                 use_perceptual=False, 
                 perceptual_weight=0.1,
                 use_ssim=False,
                 ssim_weight=0.1):
        """
        Args:
            loss_type: Base diffusion loss type
            use_perceptual: Whether to use perceptual loss
            perceptual_weight: Weight for perceptual loss
            use_ssim: Whether to use SSIM loss
            ssim_weight: Weight for SSIM loss
        """
        super().__init__()
        
        self.diffusion_loss = DiffusionLoss(loss_type=loss_type)
        
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        if use_ssim:
            self.ssim_loss = SSIMLoss()
    
    def forward(self, pred_noise, true_noise, pred_x0=None, target_x0=None):
        """
        Calculate combined loss
        
        Args:
            pred_noise: Predicted noise (B, C, H, W)
            true_noise: Ground truth noise (B, C, H, W)
            pred_x0: Predicted clean image (optional, for perceptual/SSIM loss)
            target_x0: Target clean image (optional, for perceptual/SSIM loss)
        
        Returns:
            Total loss (scalar), dict of individual losses
        """
        # Base diffusion loss
        diff_loss = self.diffusion_loss(pred_noise, true_noise)
        total_loss = diff_loss
        
        loss_dict = {'diffusion': diff_loss.item()}
        
        # Perceptual loss (if provided clean images)
        if self.use_perceptual and pred_x0 is not None and target_x0 is not None:
            perc_loss = self.perceptual_loss(pred_x0, target_x0)
            total_loss = total_loss + self.perceptual_weight * perc_loss
            loss_dict['perceptual'] = perc_loss.item()
        
        # SSIM loss (if provided clean images)
        if self.use_ssim and pred_x0 is not None and target_x0 is not None:
            ssim_loss_val = self.ssim_loss(pred_x0, target_x0)
            total_loss = total_loss + self.ssim_weight * ssim_loss_val
            loss_dict['ssim'] = ssim_loss_val.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss


class WeightedDiffusionLoss(nn.Module):
    """
    Diffusion loss with timestep-dependent weighting
    Gives more weight to important timesteps
    """
    
    def __init__(self, loss_type='mse', weight_schedule='constant', num_timesteps=1000):
        """
        Args:
            loss_type: Base loss type
            weight_schedule: 'constant', 'snr', or 'truncated_snr'
            num_timesteps: Number of diffusion timesteps
        """
        super().__init__()
        self.loss_type = loss_type
        self.weight_schedule = weight_schedule
        self.num_timesteps = num_timesteps
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            self.base_loss = nn.MSELoss(reduction='none')
        
        # Precompute weights if needed
        if weight_schedule != 'constant':
            self.register_buffer('weights', self._compute_weights())
    
    def _compute_weights(self):
        """Compute timestep-dependent weights"""
        # Linear schedule
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        
        if self.weight_schedule == 'snr':
            # Signal-to-noise ratio weighting
            snr = alpha_cumprod / (1 - alpha_cumprod)
            weights = snr / snr.max()
        elif self.weight_schedule == 'truncated_snr':
            # Truncated SNR (clips very small values)
            snr = alpha_cumprod / (1 - alpha_cumprod)
            weights = torch.clamp(snr, min=0.01, max=100.0)
            weights = weights / weights.max()
        else:
            weights = torch.ones(self.num_timesteps)
        
        return weights
    
    def forward(self, pred_noise, true_noise, timesteps):
        """
        Calculate weighted diffusion loss
        
        Args:
            pred_noise: Predicted noise (B, C, H, W)
            true_noise: Ground truth noise (B, C, H, W)
            timesteps: Timesteps for each sample (B,)
        
        Returns:
            Weighted loss (scalar)
        """
        # Calculate base loss (per sample)
        loss = self.base_loss(pred_noise, true_noise)
        loss = loss.mean(dim=[1, 2, 3])  # Average over channels and spatial dims
        
        # Apply timestep-dependent weights
        if self.weight_schedule != 'constant':
            weights = self.weights[timesteps]
            loss = loss * weights
        
        return loss.mean()


# Factory function for easy loss creation
def get_loss_function(config):
    """
    Factory function to create loss function from config
    
    Args:
        config: Dictionary with loss configuration
            - loss_type: 'mse', 'l1', 'smooth_l1', 'huber'
            - use_combined: Whether to use combined loss
            - use_perceptual: Whether to add perceptual loss
            - perceptual_weight: Weight for perceptual loss
            - use_ssim: Whether to add SSIM loss
            - ssim_weight: Weight for SSIM loss
            - use_weighted: Whether to use weighted loss
            - weight_schedule: 'constant', 'snr', 'truncated_snr'
    
    Returns:
        Loss function module
    """
    loss_type = config.get('loss_type', 'mse')
    
    if config.get('use_weighted', False):
        return WeightedDiffusionLoss(
            loss_type=loss_type,
            weight_schedule=config.get('weight_schedule', 'constant'),
            num_timesteps=config.get('num_timesteps', 1000)
        )
    elif config.get('use_combined', False):
        return CombinedDiffusionLoss(
            loss_type=loss_type,
            use_perceptual=config.get('use_perceptual', False),
            perceptual_weight=config.get('perceptual_weight', 0.1),
            use_ssim=config.get('use_ssim', False),
            ssim_weight=config.get('ssim_weight', 0.1)
        )
    else:
        return DiffusionLoss(
            loss_type=loss_type,
            use_l1=config.get('use_l1', False),
            l1_weight=config.get('l1_weight', 0.1)
        )
##########training loop##########
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np


# Import noise scheduler from your existing code
import sys
sys.path.append('.')


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.condition_encoder = ConditionEncoder(
            gridsat_channels=1,
            era5_channels=4,
            img_size=config['img_size'],
            embed_dim=config['embed_dim'],
            output_channels=config['condition_channels']
        ).to(self.device)
        
        self.unet = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        base_channels=config['base_channels'],
        channel_mults=config['channel_mults'],
        t_emb_dim=config['t_emb_dim'],
        num_heads=config['num_heads'],
        gridsat_channels=1,  # Add this
        era5_channels=4,     # Add this
        img_size=config['img_size'],  # Add this
        condition_embed_dim=config['embed_dim']  # Replace context_dim with this
        ).to(self.device)
        
        # Noise scheduler
        self.scheduler = LinearNoiseScheduler(
            num_timesteps=config['num_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end']
        )
        
        # Move scheduler tensors to device
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alpha_cumprod = self.scheduler.alpha_cumprod.to(self.device)
        self.scheduler.sqrt_alpha_cumprod = self.scheduler.sqrt_alpha_cumprod.to(self.device)
        self.scheduler.sqrt_one_minus_alpha_cumprod = self.scheduler.sqrt_one_minus_alpha_cumprod.to(self.device)
        
        # Loss function
        self.criterion = DiffusionLoss(
            loss_type=config['loss_type'],
            use_l1=config.get('use_l1', False),
            l1_weight=config.get('l1_weight', 0.1)
        )
        
        # Optimizer
        params = list(self.condition_encoder.parameters()) + list(self.unet.parameters())
        self.optimizer = AdamW(params, lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
        
        # Learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config['num_epochs'],
            eta_min=config['min_lr']
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.condition_encoder.train()
        self.unet.train()
        
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            # Move data to device
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            target = batch['target'].to(self.device)
            timestamps = batch['timestamp']
            storm_names = batch['storm_name']
            
            batch_size = target.shape[0]
            
            # Encode conditions
            encoded_cond, context_emb = self.condition_encoder(
                gridsat, era5, timestamps, storm_names
            )
            
            # Sample random timesteps
            t = torch.randint(0, self.scheduler.num_timesteps, 
                            (batch_size,), device=self.device)
            
            # Add noise to target
            noise = torch.randn_like(target)
            noisy_target = self.scheduler.add_noise(target, noise, t)
            
            # Predict noise
            noise_pred = self.unet(noisy_target, encoded_cond, t, context_emb)
            
            # Calculate loss
            loss = self.criterion(noise_pred, noise)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.condition_encoder.parameters()) + list(self.unet.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model"""
        self.condition_encoder.eval()
        self.unet.eval()
        
        epoch_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc='Validation'):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            target = batch['target'].to(self.device)
            timestamps = batch['timestamp']
            storm_names = batch['storm_name']
            
            batch_size = target.shape[0]
            
            # Encode conditions
            encoded_cond, context_emb = self.condition_encoder(
                gridsat, era5, timestamps, storm_names
            )
            
            # Sample random timesteps
            t = torch.randint(0, self.scheduler.num_timesteps,
                            (batch_size,), device=self.device)
            
            # Add noise
            noise = torch.randn_like(target)
            noisy_target = self.scheduler.add_noise(target, noise, t)
            
            # Predict noise
            noise_pred = self.unet(noisy_target, encoded_cond, t, context_emb)
            
            # Calculate loss
            loss = self.criterion(noise_pred, noise)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'condition_encoder_state_dict': self.condition_encoder.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.condition_encoder.load_state_dict(checkpoint['condition_encoder_state_dict'])
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        start_epoch = 0
        
        # Load checkpoint if resuming
        if self.config.get('resume_checkpoint'):
            start_epoch = self.load_checkpoint(self.config['resume_checkpoint']) + 1
        
        print(f"Starting training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Learning rate step
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6e}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.6f}")
            
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Generate sample images
            if (epoch + 1) % self.config['sample_every'] == 0:
                save_comparison_images(
                    self.unet,
                    val_loader,
                    self.scheduler,
                    self.condition_encoder,
                    self.device,
                    self.output_dir / f'epoch_{epoch + 1}',
                    num_samples=8
                )
            
            # Plot training curves
            if (epoch + 1) % self.config['plot_every'] == 0:
                plot_training_curves(
                    self.train_losses,
                    self.val_losses,
                    self.output_dir / 'training_curves.png'
                )
        
        # Final save
        self.save_checkpoint(self.config['num_epochs'] - 1, is_best=False)
        
        # Final evaluation
        print("\n" + "="*50)
        print("Final Evaluation")
        print("="*50)
        
        metrics = evaluate_model(
            self.unet,
            val_loader,
            self.scheduler,
            self.device,
            self.condition_encoder,
            num_samples=self.config.get('eval_samples', 100)
        )
        
        print("\nFinal Metrics:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric.upper()}: {value:.6f}")
        
        # Save metrics
        with open(self.output_dir / 'final_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nTraining complete! Results saved to {self.output_dir}")

def main():
    """Main training function"""
    
    # Configuration
    config = {
        # Data
        'root_dir': r'/kaggle/input/setcd-dataset',  
        'train_years': ['2005_0', '2016_0', '2022_0'],
        'test_storm': '2022349N13068',
        'batch_size': 8,
        'num_workers': 4,
        'img_size': 64,
        
        # Model architecture
        'base_channels': 64,
        'channel_mults': (1, 2, 4, 8),
        'num_heads': 4,
        'embed_dim': 128,
        'condition_channels': 64,
        't_emb_dim': 128,
        
        # Diffusion
        'num_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        
        # Training
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'loss_type': 'mse',  # 'mse', 'l1', or 'smooth_l1'
        'use_l1': False,
        'l1_weight': 0.1,
        
        # Checkpointing
        'checkpoint_dir': './checkpoints',
        'output_dir': './outputs',
        'save_every': 5,
        'sample_every': 5,
        'plot_every': 1,
        'eval_samples': 100,
        'resume_checkpoint': None,  # Path to checkpoint to resume from
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    print(json.dumps(config, indent=4))
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    
    train_loader, val_loader = get_train_val_dataloaders(
        root_dir=config['root_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_years=config['train_years'],
        val_years=config['train_years'],  # Uses same years but filters by test_storm
        test_storm=config['test_storm']
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config)
    
    # Start training
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
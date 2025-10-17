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
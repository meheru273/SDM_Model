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
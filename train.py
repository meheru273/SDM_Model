import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from dataloader import get_dataloader, get_test_dataloader
from unet_base import ConditionalUNet
from condition_encoding import ConditionEncoder
from loss import DiffusionLoss, CombinedDiffusionLoss
from evaluation import (evaluate_model, save_comparison_images, 
                       plot_training_curves, MetricsCalculator)

# Import noise scheduler from your existing code
import sys
sys.path.append('.')
from unet_base import LinearNoiseScheduler  # Assuming it's in the same file


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
            condition_channels=config['condition_channels'],
            base_channels=config['base_channels'],
            channel_mults=config['channel_mults'],
            t_emb_dim=config['t_emb_dim'],
            context_dim=config['embed_dim'],
            num_heads=config['num_heads']
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
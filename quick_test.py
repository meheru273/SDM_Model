
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os

# Make sure to import your model classes
# from your_training_script import DebugConditionalUNet, LinearNoiseScheduler, CycloneDataset, get_dataloader
from dataloader import get_dataloader , CycloneDataset
from unet_base import ConditionalUNet , LinearNoiseScheduler
def quick_test_model(model_path, data_root, device='auto'):
    """
    Quick test function for your cyclone forecasting model

    Args:
        model_path: Path to your .pt checkpoint file
        data_root: Path to your data directory (E:/setcd)
        device: 'auto', 'cuda', or 'cpu'
    """

    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    print(f"Data root: {data_root}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Initialize model (adjust parameters based on your training config)
    model = ConditionalUNet(
        in_channels=3,      # Adjust if you used different channels
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        t_emb_dim=128,
        num_heads=4
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Initialize noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    )

    # Move scheduler to device
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_cumprod = scheduler.alpha_cumprod.to(device)
    scheduler.sqrt_alpha_cumprod = scheduler.sqrt_alpha_cumprod.to(device)
    scheduler.sqrt_one_minus_alpha_cumprod = scheduler.sqrt_one_minus_alpha_cumprod.to(device)

    # Create test dataloader
    print("\nLoading test data...")
    test_loader = get_dataloader(
        root_dir=data_root,
        batch_size=2,  # Small batch for testing
        num_workers=0,
        shuffle=False,
        years=['2005_0'],  # Use 2022 data for testing
        data_type='GRIDSAT_data.npy',
        max_samples=10  # Just test with 10 samples for quick testing
    )

    print(f"Test dataset loaded with {len(test_loader.dataset)} samples")

    # Test the model
    print("\nRunning inference...")
    results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            if i >= 2:  # Only test first 2 batches
                break

            condition = batch['condition'].to(device)
            target = batch['target'].to(device)

            # Generate prediction using sampling
            prediction = sample_cyclone_prediction(model, scheduler, condition, device)

            # Calculate simple metrics
            mse = torch.nn.functional.mse_loss(prediction, target)
            mae = torch.nn.functional.l1_loss(prediction, target)

            results.append({
                'batch': i,
                'mse': mse.item(),
                'mae': mae.item(),
                'condition': condition.cpu(),
                'target': target.cpu(),
                'prediction': prediction.cpu()
            })

            print(f"Batch {i}: MSE={mse.item():.4f}, MAE={mae.item():.4f}")

    # Create visualizations
    print("\nCreating visualizations...")
    create_quick_visualization(results)

    # Summary
    avg_mse = np.mean([r['mse'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])

    print(f"\nTest Results Summary:")
    print(f"====================")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Visualizations saved to: test_results/")

    return results

@torch.no_grad()
def sample_cyclone_prediction(model, scheduler, condition, device, num_steps=20):
    """
    Generate cyclone prediction using the diffusion model

    Args:
        model: The trained diffusion model
        scheduler: The noise scheduler
        condition: Input condition (current cyclone state)
        device: Computing device
        num_steps: Number of denoising steps (fewer = faster)
    """
    batch_size = condition.shape[0]

    # Start with random noise
    x = torch.randn_like(condition).to(device)

    # Sample fewer timesteps for faster inference
    timesteps = torch.linspace(
        scheduler.num_timesteps - 1, 0, 
        num_steps, dtype=torch.long, device=device
    )

    for t in timesteps:
        t_batch = t.repeat(batch_size)

        # Predict noise
        noise_pred = model(x, condition, t_batch)

        # Remove noise
        x, _ = scheduler.sample_prev_timestep(x, noise_pred, t.item())

    return x

def create_quick_visualization(results):
    """Create quick visualizations of test results"""
    os.makedirs('test_results', exist_ok=True)

    for i, result in enumerate(results):
        condition = result['condition'][0, 0]  # First sample, first channel
        target = result['target'][0, 0]
        prediction = result['prediction'][0, 0]

        # Denormalize from [-1, 1] to [0, 1] for better visualization
        condition = (condition + 1) / 2
        target = (target + 1) / 2
        prediction = (prediction + 1) / 2

        # Create subplot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Condition
        im1 = axes[0].imshow(condition, cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title('Input (Time T)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Target
        im2 = axes[1].imshow(target, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth (T+3h)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Prediction
        im3 = axes[2].imshow(prediction, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('Prediction (T+3h)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        # Difference
        diff = np.abs(target.numpy() - prediction.numpy())
        im4 = axes[3].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[3].set_title('Absolute Difference')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3], fraction=0.046)

        plt.tight_layout()
        plt.savefig(f'test_results/test_sample_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(results)} visualization plots")

# Example usage
if __name__ == "__main__":
    # Configure these paths according to your setup
    MODEL_PATH = "checkpoints/checkpoint_epoch_100.pt"  # Update with your actual checkpoint path
    DATA_ROOT = "E:/setcd"  

    # Run quick test
    try:
        results = quick_test_model(MODEL_PATH, DATA_ROOT)
        print("\nTesting completed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Please check your model path and data directory")

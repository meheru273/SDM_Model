import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F


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
        
        # Additional NaN check after tensor conversion (safety measure)
        current_gridsat = torch.nan_to_num(current_gridsat, nan=0.0)
        current_era5 = torch.nan_to_num(current_era5, nan=0.0)
        next_gridsat = torch.nan_to_num(next_gridsat, nan=0.0)
        
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

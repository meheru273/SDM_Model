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
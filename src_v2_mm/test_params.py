import torch
from src_v2_mm.hetshield_baseline import TemporalConvGRUPredictor
p64 = TemporalConvGRUPredictor(in_channels=384, hidden_channels=64)
p128 = TemporalConvGRUPredictor(in_channels=384, hidden_channels=128)
n64 = sum(p.numel() for p in p64.parameters())
n128 = sum(p.numel() for p in p128.parameters())
print(f'hidden=64 params: {n64:,} ({n64/1e6:.2f}M)')
print(f'hidden=128 params: {n128:,} ({n128/1e6:.2f}M)')

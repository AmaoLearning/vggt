import torch
import time
from vggt.compression import (
    CompressionConfig,
    CompressionContext,
    TemporalStridePruning,
    FixedBandSpatialCompression,
    LocalRedundancyPairPruning,
    LowSimilaritySaliencyPruning
)

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Params
    B, H, S, P, D = 1, 16, 32, 1374, 64
    layer_idx = 8
    special_tokens = 5
    
    # Total input tokens
    L_in = S * P
    
    # Mechanisms to test
    mechanisms = {
        "A": TemporalStridePruning,
        "C1": FixedBandSpatialCompression,
        "D2": LocalRedundancyPairPruning,
        "F": LowSimilaritySaliencyPruning
    }
    
    # Input tensors
    q = torch.randn(B, H, S * P, D, device=device)
    k = torch.randn(B, H, S * P, D, device=device)
    v = torch.randn(B, H, S * P, D, device=device)
    
    ctx = CompressionContext(
        is_global=True,
        S=S,
        P=P,
        layer_idx=layer_idx,
        total_layers=24,
        special_tokens=special_tokens
    )
    
    for name, hook_cls in mechanisms.items():
        # Instantiate with default config
        config = CompressionConfig(mechanism=name)
        
        # Adjust some parameters to ensure they are runnable or interesting
        if name == "A":
            config.enable_q_compression = False # Focus on KV as requested (kv-only mentioned in query)
        
        hook = hook_cls(config)
        
        # Warmup
        try:
            _ = hook.compress(q, k, v, ctx)
        except Exception as e:
            print(f"Warmup failed for {name}: {e}")
            continue
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(5):
            qc, kc, vc, _ = hook.compress(q, k, v, ctx)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        
        avg_ms = ((end - start) / 5) * 1000
        L_out = kc.shape[2]
        reduction = (1 - L_out / (S * P)) * 100
        
        print(f"Mech {name}: {avg_ms:.3f} ms, Q:{qc.shape}, K:{kc.shape}, V:{vc.shape}, Reduc: {reduction:.1f}%")

if __name__ == "__main__":
    benchmark()

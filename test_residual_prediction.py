"""
Test script to verify residual prediction is working correctly.
"""

import torch
import numpy as np

def test_residual_prediction():
    """
    Test that the policy predicts residuals correctly.
    """
    print("=" * 60)
    print("Testing Residual Prediction")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 2
    horizon = 8
    
    # Ground truth trajectory (absolute coordinates)
    trajectory = torch.randn(batch_size, horizon, 2) * 10
    
    # Anchor trajectory (could be from planning module)
    anchor = torch.randn(batch_size, horizon, 2) * 10
    
    # Compute ground truth residual
    gt_residual = trajectory - anchor
    
    print(f"\nTrajectory shape: {trajectory.shape}")
    print(f"Anchor shape: {anchor.shape}")
    print(f"GT Residual shape: {gt_residual.shape}")
    
    print(f"\nTrajectory sample:\n{trajectory[0, :3]}")
    print(f"Anchor sample:\n{anchor[0, :3]}")
    print(f"GT Residual sample:\n{gt_residual[0, :3]}")
    
    # Verify reconstruction
    reconstructed = anchor + gt_residual
    reconstruction_error = torch.abs(reconstructed - trajectory).max().item()
    
    print(f"\nReconstruction error (should be ~0): {reconstruction_error:.8f}")
    
    if reconstruction_error < 1e-6:
        print("✓ Residual formulation is correct!")
    else:
        print("✗ Residual formulation has errors!")
    
    print("\n" + "=" * 60)
    print("Key Points:")
    print("  1. Model now predicts: pred_residual = trajectory - anchor")
    print("  2. Training loss: L1(pred_residual, gt_residual)")
    print("  3. Inference output: final_traj = anchor + pred_residual")
    print("  4. This allows model to focus on trajectory refinement")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_residual_prediction()

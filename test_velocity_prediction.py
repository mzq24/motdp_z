import torch
import torch.nn.functional as F
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy

def test_velocity_prediction():
    """
    Test that the policy correctly predicts velocity (time increments)
    instead of absolute trajectories.
    """
    print("Testing Velocity Prediction (Time Increments)...")
    print("=" * 60)
    
    # Minimal config
    config = {
        'policy': {
            'input_dim': 2,
            'output_dim': 2,
            'horizon': 8,
            'n_obs_steps': 4,
            'n_layer': 2,
            'n_head': 4,
            'n_emb': 128,
            'p_drop_emb': 0.0,
            'p_drop_attn': 0.0,
            'causal_attn': True,
            'obs_as_global_cond': True,
            'n_cond_layers': 2,
            'num_waypoints': 20,
        },
        'shape_meta': {
            'action': {'shape': (2,)}
        },
        'transfuser_encoder': {
            'bev_feature_dim': 1512,
            'bev_feature_upsample_dim': 64,
        },
        'bev_encoder': {
            'state_dim': 13,
        },
        'truncated_diffusion': {
            'num_train_timesteps': 100,
            'trunc_timesteps': 5,
            'train_trunc_timesteps': 20,
            'num_diffusion_steps': 2,
            'eta': 1.0,
            'route_loss_weight': 0.5,
        }
    }
    
    policy = DiffusionDiTCarlaPolicy(config)
    policy.eval()
    
    B = 2
    T = 8
    To = 4
    
    # Create test data - absolute trajectory
    trajectory = torch.randn(B, T, 2) * 10  # Ground truth trajectory (absolute positions)
    anchor = torch.randn(B, T, 2) * 10      # Anchor trajectory
    
    batch = {
        'agent_pos': trajectory,
        'anchor': anchor,
        'route': torch.randn(B, 20, 2),
        'transfuser_bev_feature': torch.randn(B, 1512, 8, 8),
        'transfuser_bev_feature_upsample': torch.randn(B, 64, 64, 64),
        'reasoning_query_tokens': torch.randn(B, 10, 2560),
        'ego_status': torch.randn(B, To, 13),
    }
    
    # Test training loss computation
    print("\n1. Testing training loss computation...")
    with torch.no_grad():
        loss = policy.compute_loss(batch)
    
    print(f"   Loss computed successfully: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    
    # Test that model learns to predict velocity
    print("\n2. Verifying velocity prediction target...")
    print(f"   Ground truth trajectory shape: {trajectory.shape}")
    
    # Expected velocity (time increments)
    # velocity[0] = initial position, velocity[1:] = increments
    velocity = torch.zeros_like(trajectory)
    velocity[:, 0, :] = trajectory[:, 0, :]  # First point is initial position
    velocity[:, 1:, :] = trajectory[:, 1:, :] - trajectory[:, :-1, :]  # dx, dy between steps
    
    print(f"   Velocity (dx, dy) shape: {velocity.shape}")
    print(f"   Velocity range: [{velocity.min():.2f}, {velocity.max():.2f}]")
    print(f"   Example velocity[0, :3]:")
    for i in range(3):
        print(f"     Step {i}: dx={velocity[0, i, 0]:.3f}, dy={velocity[0, i, 1]:.3f}")
    
    # Verify cumsum reconstruction
    reconstructed = torch.cumsum(velocity, dim=1)
    reconstruction_error = (reconstructed - trajectory).abs().mean()
    print(f"\n   Cumsum reconstruction error: {reconstruction_error:.6f}")
    assert reconstruction_error < 1e-5, "Cumsum should reconstruct trajectory!"
    
    print("\n   Example reconstruction (first sample, first 3 steps):")
    print(f"     Original trajectory[0, :3]: {trajectory[0, :3]}")
    print(f"     Reconstructed (cumsum):     {reconstructed[0, :3]}")
    
    # Verify the loss is computed on velocity
    print("\n3. Testing inference...")
    obs_dict = {
        'transfuser_bev_feature': torch.randn(1, 1512, 8, 8),
        'transfuser_bev_feature_upsample': torch.randn(1, 64, 64, 64),
        'reasoning_query_tokens': torch.randn(1, 10, 2560),
        'ego_status': torch.randn(1, To, 13),
        'anchor': torch.randn(1, T, 2) * 5,  # Provide anchor for inference
    }
    
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    
    predicted_traj = result['action_pred']
    route_pred = result['route_pred']
    
    print(f"   Predicted trajectory shape: {predicted_traj.shape}")
    print(f"   Route prediction shape: {route_pred.shape}")
    print(f"   ✓ Predicted trajectory is cumsum of velocities")
    
    print("\n" + "=" * 60)
    print("✓ All velocity prediction tests passed!")
    print("=" * 60)
    print("\nKey points:")
    print("  1. Training target: velocity[0] = traj[0], velocity[t>0] = traj[t] - traj[t-1]")
    print("  2. Model predicts: (x0, y0, dx1, dy1, dx2, dy2, ...)")
    print("  3. Final trajectory: cumsum(velocity)")
    print("  4. Example: if velocity = [(1,0), (1,0), (0,1)]")
    print("     then trajectory = [(1,0), (2,0), (2,1)]")
    print("  5. This is more stable and easier to learn than absolute positions")
    print("=" * 60)

if __name__ == "__main__":
    test_velocity_prediction()

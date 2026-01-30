from training.train_carla_bev import validate_model, create_carla_config
from dataset.generate_pdm_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
from torch.utils.data import DataLoader
import torch
import os


config = create_carla_config()


dataset_path_root = config.get('training', {}).get('dataset_path')
train_dataset_path = os.path.join(dataset_path_root, 'train')
val_dataset_path = os.path.join(dataset_path_root, 'val')
image_data_root = config.get('training', {}).get('image_data_root')
train_dataset = CARLAImageDataset(dataset_path=train_dataset_path, image_data_root=image_data_root)
val_dataset = CARLAImageDataset(dataset_path=val_dataset_path, image_data_root=image_data_root)

# 增加 num_workers 并将 batch_size 减小以便测试
batch_size=64
num_workers=8 # 增加 a more reasonable value, e.g., 4 or 8
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


action_stats = {
    'min': torch.tensor([0, -10.5050]),
    'max': torch.tensor([24.4924,  9.9753]),
    'mean': torch.tensor([2.3079, 0.0188]),
    'std': torch.tensor([3.7443, 0.6994]),
}
policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats)


from time import time
policy.train()
policy.cuda()
lr = config.get('optimizer', {}).get('lr', 5e-5)
weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

start_time = time()
max_batch = 1024
with torch.no_grad():
    for batch_idx, batch in enumerate(val_dataset):
        if batch_idx >= max_batch:
            break
    t1 = time()
    print("dataloader time:", t1 - start_time)
        # for key in batch:
        #     if isinstance(batch[key], torch.Tensor):
        #         batch[key] = batch[key].to("cuda")
        #t2 = time()
        #print("to cuda time:", t2 - t1)
        #for i in range(1):
        #    pass
            #loss = policy.compute_loss(batch)
        #t3 = time()
        #print("forward time:", t3 - t2)
        
print("Time:", (time() - start_time))
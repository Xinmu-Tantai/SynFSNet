

from torch.utils.data import DataLoader
from synfs.losses import *
from synfs.datasets.vaihingen_dataset import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import os

# Create path variables with os.path to ensure cross-platform compatibility
base_dir = os.path.abspath("SynFSNet-main")
data_dir = os.path.join(base_dir, "data")
weights_dir = os.path.join(base_dir, "model_weights")
logs_dir = os.path.join(base_dir, "lightning_logs")

# Create directories if they don't exist
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size =4
val_batch_size = 4
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

# Update naming scheme to include timestamp for uniqueness
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
weights_name = f"synfsnet-head88-bz{train_batch_size}-down2"


# Path configurations
weights_path = os.path.join(weights_dir, "vaihingen", weights_name)
log_name = os.path.join(logs_dir, "vaihingen", weights_name)
test_weights_name = "weights_name"
# Ensure directories exist
os.makedirs(os.path.join(weights_dir, "vaihingen"), exist_ok=True)
os.makedirs(os.path.join(logs_dir, "vaihingen"), exist_ok=True)

# Training monitoring configuration
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k =-1  # Save more checkpoints to track progress
save_last = True
check_val_every_n_epoch = 1
gpus = [0]  # Using a single GPU

# Checkpoint paths
pretrained_ckpt_path = None
resume_ckpt_path = None

# define the network
from synfs.model.synfsnet import SynFSNet
net = SynFSNet(n_class=num_classes)

#define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
use_aux_loss = False

# Define data paths with os.path.join for platform independence
train_data_root = os.path.join(data_dir, "vaihingen", "train")
test_data_root = os.path.join(data_dir, "vaihingen", "test")

# define the dataloader
train_dataset = VaihingenDataset(data_root=train_data_root, mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root=test_data_root, transform=val_aug)
test_dataset = VaihingenDataset(data_root=test_data_root, transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=15,
    T_mult=2,
    eta_min=1e-6
)

# Print configuration for verification
print(f"Model weights will be saved to: {weights_path}")
print(f"Logs will be saved to: {log_name}")
print(f"Using {max_epoch} epochs with batch size {train_batch_size}")
print(f"Monitoring {monitor} in {monitor_mode} mode, saving top {save_top_k} models")

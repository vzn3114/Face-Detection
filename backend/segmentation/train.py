import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

from segmentation.config import (
    MODEL_WEIGHTS_PATH, NUM_CLASSES, BATCH_SIZE, 
    LEARNING_RATE, NUM_EPOCHS, DEVICE, CROP_SIZE
)
from segmentation.model import UNet
from segmentation.dataset import FaceMask

def train(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize Dataset
    print("Loading CelebAMask-HQ Dataset...")
    full_dataset = FaceMask(cropsize=CROP_SIZE, mode='train')
    
    if args.debug:
        print("DEBUG MODE: Training on a small subset of 20 images...")
        subset_indices = list(range(min(20, len(full_dataset))))
        dataset = Subset(full_dataset, subset_indices)
        batch_size = 4
        epochs = 2
    else:
        dataset = full_dataset
        batch_size = args.batch_size
        epochs = args.epochs
        
    print(f"Total training samples: {len(dataset)}")
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if args.debug else 4,
        drop_last=True
    )
    
    # Initialize Model
    print("Initializing U-Net model...")
    model = UNet(n_channels=3, n_classes=NUM_CLASSES)
    model.to(device)
    
    # Optimizer & Loss Function
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Ensure weights parent directory exists
    weights_dir = os.path.dirname(MODEL_WEIGHTS_PATH)
    if weights_dir and not os.path.exists(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
        
    print("Starting training loop...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Target has shape (B, 1, H, W) -> squeeze to (B, H, W)
            targets = targets.squeeze(1).long()
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)  # shape (B, 19, H, W)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {epoch_loss:.4f}")
        
        # Save model if it has the best loss or if in debug/last epoch
        if epoch_loss < best_loss or epoch == epochs - 1:
            best_loss = epoch_loss
            print(f"Saving model weights to {MODEL_WEIGHTS_PATH}...")
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            
    print("Training finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Face Segmentation on CelebAMask-HQ")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cuda or cpu)")
    parser.add_argument("--debug", action="store_true", help="Run a quick debug training run on CPU")
    args = parser.parse_args()
    
    train(args)

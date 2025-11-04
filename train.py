import torch
from torch import nn
import torch.optim as optim
import wandb # Tool for tracking results (as required by Lab 3)

# Import local modules from the project structure
from models.CustomNet import CustomNet
# Import the loaders directly from the dataset module
from dataset.tinyimagenet_dataloader import train_loader, val_loader 
from eval import validate # Import validation logic

# --- Configuration for Wandb (Lab 3 requirement) ---
WANDB_PROJECT_NAME = "mldl-tinyimagenet-project"
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# --- Training Function for One Epoch ---
def train(epoch, model, train_loader, criterion, optimizer, device):
    """
    Performs one epoch of training on the provided model.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move tensors to the designated device (GPU or CPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Optimization Step
        optimizer.zero_grad() # Reset gradients
        outputs = model(inputs)
        loss = criterion(outputs, targets) # Compute loss
        
        # Backpropagation
        loss.backward() 
        optimizer.step() 

        # Statistics Update
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Calculate epoch statistics
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    # Log metrics to Wandb (Lab 3 requirement)
    if wandb.run:
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "epoch": epoch
        })


# --- Main Execution Loop ---
if __name__ == '__main__':
    
    # 1. Device and Wandb Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Wandb run for result tracking (Lab 3 requirement)
    wandb.init(project=WANDB_PROJECT_NAME, config={
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "batch_size": train_loader.batch_size if train_loader else None,
        "architecture": "CustomNet",
        "dataset": "TinyImageNet"
    })

    # 2. Model Initialization (Putting everything together)
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Log model information to Wandb
    if wandb.run:
        wandb.watch(model, criterion, log="all", log_freq=10)


    best_acc = 0
    if train_loader is None or val_loader is None:
        print("Cannot start training. Please check dataset loading errors above.")
    else:
        # 3. Start Training Loop
        print("Starting Training...")
        for epoch in range(1, NUM_EPOCHS + 1):
            
            # Train step
            train(epoch, model, train_loader, criterion, optimizer, device)

            # Validation step
            val_accuracy = validate(model, val_loader, criterion) 
            
            # Log validation metrics
            if wandb.run:
                 wandb.log({"val_accuracy": val_accuracy, "epoch": epoch})

            # Checkpoint the best model (best practice)
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                # Save model to 'checkpoints/' folder (part of the project skeleton)
                torch.save(model.state_dict(), 'checkpoints/best_customnet.pth')
                print(f"Model saved to checkpoints/best_customnet.pth with improved accuracy: {best_acc:.2f}%")

        print(f'Final Best validation accuracy: {best_acc:.2f}%')
        wandb.finish()
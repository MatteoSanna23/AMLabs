import torch
from torch import nn
import torch.optim as optim
import yaml
# import wandb # Tool for tracking results (Lab 3) -- remains commented, configurable

# Import local modules
from models.customNet import CustomNet
from dataset.tinyimagenet_dataloader import train_loader, val_loader
from eval import validate

# --- Load configuration ---
try:
    # Default config from repo
    config_path = "config_local.yaml"  # Colab override if exists
    config = yaml.safe_load(open(config_path))
except FileNotFoundError:
    config_path = "config.yaml"
    config = yaml.safe_load(open(config_path))

# Override parameters locally if needed (example for Colab)
# config_override = {"num_workers": 2, "batch_size": 64, "epochs": 1}
# config.update(config_override)

# Extract config variables
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
IMAGE_SIZE = config["image_size"]
LEARNING_RATE = config["learning_rate"]
MOMENTUM = config["momentum"]
NUM_EPOCHS = config["epochs"]
DATA_ROOT = config["data_root"]

# --- Configuration for Wandb ---
WANDB_PROJECT_NAME = config.get("wandb_project_name", "mldl-tinyimagenet-project")

# --- Training Function ---
def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    # Log to Wandb if enabled
    # if wandb.run:
    #     wandb.log({
    #         "train_loss": train_loss,
    #         "train_accuracy": train_accuracy,
    #         "epoch": epoch
    #     })

# --- Main ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Wandb run
    # wandb.init(project=WANDB_PROJECT_NAME, config={
    #     "learning_rate": LEARNING_RATE,
    #     "momentum": MOMENTUM,
    #     "batch_size": BATCH_SIZE,
    #     "architecture": "CustomNet",
    #     "dataset": "TinyImageNet"
    # })

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # wandb.watch(model, criterion, log="all", log_freq=10)

    best_acc = 0
    if train_loader is None or val_loader is None:
        print("Cannot start training. Please check dataset loading errors.")
    else:
        print("Starting Training...")
        for epoch in range(1, NUM_EPOCHS + 1):
            train(epoch, model, train_loader, criterion, optimizer, device)
            val_accuracy = validate(model, val_loader, criterion)

            # Log validation metrics
            # if wandb.run:
            #     wandb.log({"val_accuracy": val_accuracy, "epoch": epoch})

            # Save best model
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                torch.save(model.state_dict(), 'checkpoints/best_customnet.pth')
                print(f"Model saved with improved accuracy: {best_acc:.2f}%")

        print(f'Final Best validation accuracy: {best_acc:.2f}%')
        # wandb.finish()

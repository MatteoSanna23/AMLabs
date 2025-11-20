# train.py
def train_model(cfg):
    import torch
    from torch import nn, optim
    import yaml
    from models.customNet import CustomNet
    from models.alexNet import AlexNet
    from dataset.tinyimagenet_dataloader import train_loader, val_loader
    from eval import validate
    # --- Device ---
    device = "cuda" if torch.cuda.is_available() and cfg.get("device") != "cpu" else "cpu"
    print(f"Using device: {device}")

    # --- Model ---
    model_name = cfg.get("model", "customnet").lower()  # normalizza a minuscolo

    if model_name == "customnet":
        model = CustomNet().to(device)
    elif model_name == "alexnet":
        model = AlexNet(num_classes=200).to(device)
    else:
        raise ValueError(f"Model '{model_name}' not recognized. Use 'customnet' or 'alexnet'.")

    # --- Optimizer ---
    optimizer_name = cfg.get("optimizer", "adam").lower()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=cfg.get("momentum", 0.9))
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # --- Loss Function and Epochs ---
    criterion = nn.CrossEntropyLoss()
    epochs = cfg.get("epochs", 5)

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)   # Move to device

            optimizer.zero_grad()   # Zero the parameter gradients
            outputs = model(images) # Pass the inputs through the model
            loss = criterion(outputs, labels) # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item() # Accumulate loss, loss.item() gives the scalar value

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Validation
        val_acc = validate(model, val_loader, criterion)

        # Write Validation Accuracy
        with open("last_val_acc.txt", "w") as f:
            f.write(str(val_acc))

# This part allows the script to be run directly
if __name__ == "__main__":
    # Command line argument parsing for config file
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_local.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_model(cfg)

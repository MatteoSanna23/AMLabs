def train_model(cfg):
    import os
    import torch
    from torch import nn, optim
    from models.customNet import CustomNet
    from models.alexNet import AlexNet
    from dataset.tinyimagenet_dataloader import get_tiny_imagenet_loaders
    from eval import validate

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "") != "cpu" else "cpu"
    print(f"Using device: {device}")

    # --- Data Loaders ---
    train_loader, val_loader = get_tiny_imagenet_loaders(cfg)
    print(f"Train loader batches: {len(train_loader)} and Val loader batches: {len(val_loader)}")

    # --- Model ---
    model_name = cfg.get("model", "customnet").lower()
    if model_name == "customnet":
        model = CustomNet().to(device)
    elif model_name == "alexnet":
        model = AlexNet(num_classes=cfg.get("num_classes", 200)).to(device)
    else:
        raise ValueError(f"Model '{model_name}' not recognized. Use 'customnet' or 'alexnet'.")

    # --- Optimizer ---
    optimizer_name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("learning_rate", 1e-3)
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg.get("momentum", 0.9))
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Loss Function and Epochs ---
    criterion = nn.CrossEntropyLoss()
    epochs = cfg.get("epochs", 5)
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_acc = -float("inf")

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader) if len(train_loader) > 0 else 1
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Validation
        val_acc = validate(model, val_loader, criterion)

        # ---- SAVE LAST CHECKPOINT ----
        last_ckpt_path = os.path.join(checkpoint_dir, "last.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc
        }, last_ckpt_path)

        # write last val acc to file for easy retrieval
        with open(os.path.join(checkpoint_dir, "last_val_acc.txt"), "w") as f:
            f.write(str(val_acc))

        # ---- SAVE BEST CHECKPOINT ----
        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt_path = os.path.join(checkpoint_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc
            }, best_ckpt_path)
            print(f">>> Saved BEST checkpoint (acc={val_acc:.2f})")

    return val_acc


# This part allows the script to be run directly
if __name__ == "__main__":
    # Command line argument parsing for config file
    import argparse
    import yaml
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_local.yaml",
                        help="Prefer local config (e.g. uploaded in Colab). Fallback to config.yaml in repo.")
    args = parser.parse_args()

    # prefer config_local.yaml (for Colab) otherwise fallback to repo config.yaml
    config_path = args.config if os.path.exists(args.config) else "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found. Tried '{args.config}' and '{config_path}'")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_model(cfg)
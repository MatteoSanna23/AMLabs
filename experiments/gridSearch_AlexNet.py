import yaml
from train import train_model

def grid_search(cfg):

    # Also here, take the parameters from cfg, otherwise use default values
    learning_rates = cfg.get("grid_learning_rates", [0.1, 0.001, 0.0001])
    batch_sizes    = cfg.get("grid_batch_sizes",    [16, 32, 64])

    RESULTS = []

    for lr in learning_rates:
        for bs in batch_sizes:

            cfg["learning_rate"] = lr
            cfg["batch_size"] = bs

            # (optional) generate config_local.yaml
            with open("config_local.yaml", "w") as f:
                yaml.dump(cfg, f)

            print("\n>> config_local.yaml generated:", cfg)
            print(f"--> TRAINING LR={lr} BS={bs}")

            # Train
            val_acc = train_model(cfg)

            # Save results
            RESULTS.append({
                "lr": lr,
                "batch_size": bs,
                "val_acc": val_acc
            })

            print(f"=== Result: LR={lr} BS={bs} -> {val_acc:.2f}% ===")

    # Best result
    best = max(RESULTS, key=lambda x: x["val_acc"])

    print("\n=========== FINAL RESULTS ===========")
    for r in RESULTS:
        print(r)

    print("\n>>> BEST CONFIG <<<")
    print(best)

    return best

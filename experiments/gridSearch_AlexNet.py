import yaml
from train import train_model
import gc
import torch

def grid_search(cfg):

    # Also here, take the parameters from cfg, otherwise use default values
    learning_rates = cfg.get("grid_learning_rates", [0.1, 0.001, 0.0001])
    batch_sizes    = cfg.get("grid_batch_sizes",    [16, 32, 64])

    RESULTS = []

    for lr in learning_rates:
        for bs in batch_sizes:
            
            print(f"\n--> TRAINING LR={lr} | BS={bs}")

            # Work in a shallow copy of cfg
            run_cfg = dict(cfg)
            run_cfg["learning_rate"] = lr
            run_cfg["batch_size"] = bs
            run_cfg["model"] = "alexnet"
            run_cfg["epochs"] = cfg.get("epochs", 1)


            # Train
            val_acc = train_model(run_cfg)
            # Save results
            RESULTS.append({
                "lr": lr,
                "batch_size": bs,
                "val_acc": val_acc
            })

            print(f"=== Result: LR={lr} BS={bs} -> {val_acc:.2f}% ===")
            
            del run_cfg
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Best result
    best = max(RESULTS, key=lambda x: x["val_acc"])

    print("\n=========== FINAL RESULTS ===========")
    for r in RESULTS:
        print(r)

    print("\n>>> BEST CONFIG <<<")
    print(best)

    return best
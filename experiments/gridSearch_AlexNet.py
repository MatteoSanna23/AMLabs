import yaml
from copy import deepcopy
from train import train_model  # Importiamo la funzione direttamente

# Parametri di grid search
learning_rates = [0.1, 0.001, 0.0001]
batch_sizes = [16, 32, 64]

RESULTS = []

# Carica config originale
with open("config.yaml") as f:
    base_cfg = yaml.safe_load(f)

# Loop grid search
for lr in learning_rates:
    for bs in batch_sizes:
        temp_cfg = deepcopy(base_cfg)
        temp_cfg["learning_rate"] = lr
        temp_cfg["batch_size"] = bs

        print(f"--> Lancio train LR={lr} BS={bs}")
        val_acc = train_model(temp_cfg)  # punto 2: qui ottieni direttamente l'accurancy

        RESULTS.append({
            "lr": lr,
            "batch_size": bs,
            "val_acc": val_acc
        })

        print(f"=== Risultato: LR={lr} BS={bs} -> {val_acc:.2f}% ===")

# Stampa risultati finali
best = max(RESULTS, key=lambda x: x["val_acc"])
print("\n=========== RISULTATI FINALI ===========\n")
for r in RESULTS:
    print(r)
print("\n>>> MIGLIORE CONFIG <<<")
print(best)

import os
import yaml
from utils import data_prep  # ora prepare_val_set esiste!

REPO_URL = "https://github.com/MatteoSanna23/AMLabs.git"
REPO_DIR = "AMLabs"
DATASET_DIR = "tiny-imagenet/tiny-imagenet-200"
ZIP_PATH = "tiny-imagenet-200.zip"

# -------------------------------------------------------------
# 1. CLONA LA REPO SOLO SE NON ESISTE
# -------------------------------------------------------------
if not os.path.exists(REPO_DIR):
    print(">> Clonazione della repository...")
    os.system(f"git clone {REPO_URL}")  # chiamata minima a shell
else:
    print(">> Repository già presente, niente clone.")

os.chdir(REPO_DIR)

# -------------------------------------------------------------
# 2. INSTALLA I REQUIREMENTS
# -------------------------------------------------------------
print(">> Installazione requirements...")
os.system("pip install -r requirements.txt")

# -------------------------------------------------------------
# 3. CARICA IL CONFIG E CREA IL CONFIG_LOCALE
# -------------------------------------------------------------
print(">> Generazione config_local.yaml...")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Override per Colab
override_cfg = {
    "num_workers": 2,
    "batch_size": 32,
    "learning_rate": 0.001,
}
cfg.update(override_cfg)

with open("config_local.yaml", "w") as f:
    yaml.dump(cfg, f)

print(">> config_local.yaml generato:", cfg)

# -------------------------------------------------------------
# 4. DATA PREP — riorganizza validation set
# -------------------------------------------------------------
print(">> Preparazione dataset (riorganizzazione validation set)...")
data_prep.prepare_val_set()  # ora funziona correttamente

print(">> SETUP COMPLETATO! Ora puoi lanciare train_model(cfg) o grid search direttamente dal notebook")

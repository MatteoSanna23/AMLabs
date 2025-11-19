import os
import shutil

# Comandi bash per il download (questi sono solitamente eseguiti in Colab/Terminal)
# Dovresti eseguirli manualmente la prima volta
def download_and_extract_tiny_imagenet():
    print("Esecuzione download/unzip. Potrebbe essere necessario eseguirlo manualmente in bash/colab.")
    # Esegui in bash:
    # !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    # !unzip tiny-imagenet-200.zip -d tiny-imagenet

def reorganize_val_set():
    """
    Riformatta la cartella 'val' per essere compatibile con ImageFolder.
    Crea sottocartelle per ogni classe e sposta le immagini.
    """
    val_annotations_path = 'tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt'
    base_val_path = 'tiny-imagenet/tiny-imagenet-200/val/'
    images_path = os.path.join(base_val_path, 'images')

    if not os.path.exists(val_annotations_path):
        print("Errore: val_annotations.txt non trovato. Assicurati di aver scaricato ed estratto il dataset.")
        return

    with open(val_annotations_path) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            # Crea la sottocartella per la classe (se non esiste)
            os.makedirs(os.path.join(base_val_path, cls), exist_ok=True)

            # Sposta l'immagine nella sua cartella di classe
            src = os.path.join(images_path, fn)
            dst = os.path.join(base_val_path, cls, fn)
            shutil.copyfile(src, dst)

    # Rimuovi la cartella originale 'images'
    shutil.rmtree(images_path)
    print("Riorganizzazione del validation set completata.")

def prepare_val_set():
    """
    Funzione unica per setup script:
    - eventualmente scarica dataset
    - riorganizza la validation set
    """
    # download_and_extract_tiny_imagenet()  # opzionale se vuoi gestire tutto in Python
    reorganize_val_set()

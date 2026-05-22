# Classificazione Egocart con Deep Learning

Progetto di Deep Learning per la classificazione di immagini egocentriche acquisite da carrelli della spesa, utilizzando reti neurali convoluzionali implementate in PyTorch.

## Risultati

| Modello | Test Acc | Macro F1 | Parametri | GPU (MB) | Inf. (ms/img) |
|---|---|---|---|---|---|
| Custom CNN | 93.4% | 0.935 | 459K | 1.086 | 4.7 |
| MobileNetV2 (fine-tuning) | **96.2%** | **0.963** | 2.2M | 2.473 | 9.3 |
| MobileNetV2 (feature extraction) | 91.3% | 0.920 | 2.2M | 365 | 9.3 |

## Installazione e Setup

### Prerequisiti

- **Python 3.10** installato sul sistema

### Dataset

Scaricare il dataset EgoCart dal [sito ufficiale](https://iplab.dmi.unict.it/legacy/EgocentricShoppingCartLocalization/index.html) e posizionarlo nella cartella `egocart/` nella root del progetto.

### 1. Clonare il Repository

```bash
git clone https://github.com/alessandrofloris/EgoCartClassification
cd EgoCartClassification
```

### 2. Creazione e attivazione dell'ambiente virtuale

Su Linux/macOS:
```bash
python3.10 -m venv egocart_env
source egocart_env/bin/activate
```

Su Windows:
```bash
python -m venv egocart_env
egocart_env\Scripts\activate.bat
```

### 3. Installazione delle dipendenze

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Utilizzo

### Training

Addestramento della rete custom:
```bash
python main.py train --model CustomCNN
```

Transfer learning con MobileNetV2 (fine-tuning completo):
```bash
python main.py train --model MobileNetv2
```

Transfer learning con MobileNetV2 (feature extraction, backbone congelato):
```bash
python main.py train --model MobileNetv2 --freeze_backbone
```

### Valutazione

Valutazione sul test set con il miglior checkpoint:
```bash
python main.py eval --model CustomCNN
python main.py eval --model MobileNetv2
```

Valutazione con un checkpoint specifico:
```bash
python main.py eval --model CustomCNN --checkpoint path/to/checkpoint.pth
python main.py eval --model MobileNetv2 --checkpoint path/to/checkpoint.pth
```

### Interpretabilità (Grad-CAM)

Visualizzazione delle mappe di attivazione tramite Grad-CAM per analizzare quali regioni dell'immagine influenzano maggiormente le predizioni del modello:

```bash
python gradcam.py --model CustomCNN --checkpoint path/to/best_model_customcnn.pth
python gradcam.py --model MobileNetv2 --checkpoint path/to/best_model_mobilenetv2.pth
```

Opzioni disponibili:
- `--num_correct N`: numero di predizioni corrette da visualizzare (default: 3)
- `--num_wrong N`: numero di predizioni errate da visualizzare (default: 3)
- `--output_dir DIR`: cartella di output per le immagini generate (default: `gradcam_results/`)

Le immagini generate mostrano, per ogni campione, l'immagine originale affiancata dalla heatmap Grad-CAM sovrapposta.

## Struttura del Progetto

```
├── main.py          # Entry point: gestione argomenti e orchestrazione
├── model.py         # Definizione della rete custom (CustomCNN)
├── dataset.py       # Dataset PyTorch e trasformazioni
├── train.py         # Ciclo di training
├── eval.py          # Valutazione e metriche
├── utils.py         # Funzioni di utilità (pesi classi, confusion matrix, ecc.)
├── gradcam.py       # Analisi di interpretabilità con Grad-CAM
└── requirements.txt # Dipendenze del progetto
```

#!/usr/bin/env bash
# Se ejecuta DENTRO de la EC2 (por ssh). Prepara el repo y lanza los entrenamientos.
# Prerrequisitos en la EC2: git, python3, driver NVIDIA (viene en la DL AMI).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/alvarorf/pds-proyecto-tema1}"
BRANCH="${BRANCH:-feature/entrega2-eda-modelos}"
WORKDIR="${WORKDIR:-$HOME/pds-proyecto-tema1}"

if [[ ! -d "$WORKDIR/.git" ]]; then git clone "$REPO_URL" "$WORKDIR"; fi
cd "$WORKDIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt
# torch con CUDA: la DL AMI ya trae CUDA; si el wheel por defecto no ve GPU:
python -c "import torch;print('CUDA:',torch.cuda.is_available())" || \
  pip install torch --index-url https://download.pytorch.org/whl/cu121

# datos versionados: baja data/raw/*.csv|jsonl (standardizados) del bucket
dvc pull -v

export MLFLOW_TRACKING_URI="http://localhost:5000"

# EDA (rapido). Usa los CSV enriquecidos originales (van en git, en dataset/).
python scripts/eda_entrega2.py --data_dir dataset --out_dir docs/eda_entrega2

# v1 baseline (CPU, ~1 min)
python scripts/train.py --stage v1 --data_dir data/raw

# v2 iteracion intermedia (GPU ~20-40 min / CPU ~2-3 h). NO optimizado a proposito.
python scripts/train.py --stage v2 --data_dir data/raw \
  --model allenai/scibert_scivocab_uncased --epochs 3 --batch 16 --lr 2e-5 --max_len 256

# analisis de metricas / overfitting -> docs/MODELOS_ENTREGA2.md (tablas) + figuras
python scripts/analyze_metrics.py --results docs/model_results.json --out docs/MODELOS_ENTREGA2.md

echo ">> Listo. Revisa MLflow y haz commit de docs/ + mlflow screenshots."

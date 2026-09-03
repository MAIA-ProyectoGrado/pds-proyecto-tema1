#!/usr/bin/env bash
# Se ejecuta DENTRO de la EC2 (por ssh). Prepara el repo y lanza los entrenamientos.
# Entorno: AWS Academy Learner Lab, instancia t3.large (2 vCPU, sin GPU).
# Prerrequisitos en la EC2: git, python3, y credenciales S3 via LabInstanceProfile.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/alvarorf/pds-proyecto-tema1}"
BRANCH="${BRANCH:-feature/entrega2-eda-modelos}"
WORKDIR="${WORKDIR:-$HOME/pds-proyecto-tema1}"

sudo apt-get update -y && sudo apt-get install -y python3-venv git

if [[ ! -d "$WORKDIR/.git" ]]; then git clone "$REPO_URL" "$WORKDIR"; fi
cd "$WORKDIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt
# torch: rueda CPU (no hay GPU en el lab)
pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch

# datos versionados: baja data/raw/*.csv|jsonl (estandarizados) del bucket S3
dvc pull -v

export MLFLOW_TRACKING_URI="http://localhost:5000"
export SCIF_SKIP_MODEL_LOGGING=0

# EDA (rapido). Usa los CSV enriquecidos originales (van en git, en dataset/).
python scripts/eda_entrega2.py --data_dir dataset --out_dir docs/eda_entrega2

# v1 baseline (CPU, ~10 s)
python scripts/train.py --stage v1 --data_dir data/raw

# v2 iteracion intermedia: DistilBERT, submuestreo, 1 epoca (CPU 2 vCPU ~1.5-3 h).
# NO optimizado a proposito -> deja margen para v3.
python scripts/train.py --stage v2 --data_dir data/raw \
  --model distilbert-base-uncased --max_train 5000 --epochs 1 --max_len 128 --batch 16 --lr 3e-5

# analisis de metricas / overfitting -> docs/MODELOS_ENTREGA2.md (tablas)
python scripts/analyze_metrics.py --results docs/model_results.json --out docs/MODELOS_ENTREGA2.md

echo ">> Listo. Revisa MLflow (http://<IP>:5000), toma los pantallazos y haz commit de docs/."
echo ">> Cuando termines: ./infra/stop_ec2.sh desde tu maquina local."

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Se ejecuta DENTRO de la instancia GPU (Deep Learning AMI, CUDA ya presente).
# Entrena la iteración v3 sobre el dataset completo (14.461) y la registra en MLflow.
#
# Prerrequisitos en la instancia:
#   - MLflow escuchando en localhost:5000 (lo deja el user-data de launch_gpu_ec2.sh)
#   - ~/.aws con permiso de lectura del bucket DVC (via IAM role scif-ec2-s3)
#   - este repo clonado y la rama con la v3 del dataset
#
# Uso:
#   export REPO_URL=https://github.com/alvarorf/pds-proyecto-tema1
#   export BRANCH=feature/entrega2-eda-modelos
#   bash train_v3_gpu.sh
# ---------------------------------------------------------------------------
set -euxo pipefail

REPO_URL="${REPO_URL:-https://github.com/alvarorf/pds-proyecto-tema1}"
BRANCH="${BRANCH:-feature/entrega2-eda-modelos}"
WORK="${WORK:-$HOME/pds-proyecto-tema1}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

[ -d "$WORK/.git" ] || git clone "$REPO_URL" "$WORK"
cd "$WORK"
git fetch origin && git checkout "$BRANCH" && git pull --ff-only origin "$BRANCH"

python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-train.txt
python -c "import torch;assert torch.cuda.is_available(),'NO GPU';print('GPU:',torch.cuda.get_device_name(0))"

# datos v3 (DVC remoto 's2' = s3://scif-dvcstore-779319895642-use1)
dvc pull -r s2 dataset.dvc data/raw.dvc
python scripts/prepare_dataset.py --src dataset --out_raw data/raw --out_proc data/processed

# EDA v3
python scripts/eda_entrega2.py --data_dir dataset --out_dir docs/eda_entrega2

# --- Modelos ---------------------------------------------------------------
# v1 (baseline sobre datos v3, referencia)
python scripts/train.py --stage v1 --data_dir data/raw

# v3a: SciBERT fine-tune, datos completos, GPU
python scripts/train.py --stage v2 --data_dir data/raw --v2_method finetune \
  --model allenai/scibert_scivocab_uncased \
  --epochs 4 --batch 32 --lr 2e-5 --max_len 256 --eval_steps 100

# v3b: SPECTER2 (comparativa; comentar si falta cuota de tiempo)
python scripts/train.py --stage v2 --data_dir data/raw --v2_method finetune \
  --model allenai/specter2_base \
  --epochs 4 --batch 32 --lr 2e-5 --max_len 256 --eval_steps 100 || true

python scripts/analyze_metrics.py --results docs/model_results.json --out docs/MODELOS_ENTREGA2.md
touch "$HOME/V3_DONE"
echo ">> v3 OK. Revisa MLflow, toma pantallazos y baja docs/ + mlflow.db."
echo ">> Al terminar: aws ec2 stop-instances --instance-ids <ID>   (detener, NO terminar)."

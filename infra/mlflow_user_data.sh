#!/bin/bash
# user-data para la EC2 (Ubuntu 22.04 / Deep Learning OSS Nvidia AMI).
# Deja MLflow escuchando en el puerto 5000 con backend sqlite y artefactos locales.
set -eux

MLFLOW_DIR=/opt/mlflow
mkdir -p "$MLFLOW_DIR/artifacts"
cd "$MLFLOW_DIR"

# Python / pip
if ! command -v python3 >/dev/null; then apt-get update -y && apt-get install -y python3 python3-venv python3-pip; fi
python3 -m venv "$MLFLOW_DIR/venv"
"$MLFLOW_DIR/venv/bin/pip" install --upgrade pip
"$MLFLOW_DIR/venv/bin/pip" install "mlflow==3.15.2" boto3

cat >/etc/systemd/system/mlflow.service <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=${MLFLOW_DIR}
ExecStart=${MLFLOW_DIR}/venv/bin/mlflow server \\
  --backend-store-uri sqlite:///${MLFLOW_DIR}/mlflow.db \\
  --artifacts-destination ${MLFLOW_DIR}/artifacts \\
  --host 0.0.0.0 --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

chown -R ubuntu:ubuntu "$MLFLOW_DIR"
systemctl daemon-reload
systemctl enable --now mlflow.service

# Auto-apagado de seguridad: detiene la instancia tras 14 h de uptime (techo de costo).
cat >/etc/cron.d/auto-stop <<'EOF'
0 * * * * root bash -c 'UP=$(awk "{print int(\$1/3600)}" /proc/uptime); if [ "$UP" -ge 14 ]; then /sbin/shutdown -h now "auto-stop: 14h de uptime"; fi'
EOF

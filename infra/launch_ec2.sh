#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Lanza UNA instancia EC2 que hace de servidor MLflow + maquina de entrenamiento.
# Presupuesto: quedan ~USD 20 y como maximo 15 h de uso.
#
#   Region        : us-east-1   (bucket citation-dvcstore-tema1 esta ahi)
#   Tipo por defecto: g5.xlarge  (1x A10G 24GB)  ~USD 1.006/h  on-demand
#   Alternativa GPU : g4dn.xlarge (1x T4 16GB)   ~USD 0.526/h  (mas barata, mas lenta)
#   Fallback CPU    : c7i.2xlarge (8 vCPU)       ~USD 0.357/h  (si no hay cuota de G)
#
# Uso:
#   ./infra/launch_ec2.sh                 # g5.xlarge on-demand
#   INSTANCE_TYPE=g4dn.xlarge ./infra/launch_ec2.sh
#   INSTANCE_TYPE=c7i.2xlarge USE_GPU_AMI=0 ./infra/launch_ec2.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
USE_GPU_AMI="${USE_GPU_AMI:-1}"
KEY_NAME="${KEY_NAME:-scif-entrega2}"
SG_NAME="${SG_NAME:-scif-mlflow-sg}"
VOLUME_GB="${VOLUME_GB:-120}"
TAG="${TAG:-scif-entrega2}"
NAME_TAG="${NAME_TAG:-scif-mlflow-train}"

echo ">> Region=$REGION  Tipo=$INSTANCE_TYPE"
MY_IP="$(curl -s https://checkip.amazonaws.com || echo 0.0.0.0)"
echo ">> Tu IP publica: $MY_IP  (se usara para restringir SSH y 5000)"

# --- AMI --------------------------------------------------------------------
if [[ "$USE_GPU_AMI" == "1" ]]; then
  # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
  AMI_ID="$(aws ssm get-parameters --region "$REGION" \
    --names /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
    --query 'Parameters[0].Value' --output text)"
else
  # Ubuntu 22.04 LTS estandar
  AMI_ID="$(aws ssm get-parameters --region "$REGION" \
    --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query 'Parameters[0].Value' --output text)"
fi
echo ">> AMI: $AMI_ID"

# --- Key pair ---------------------------------------------------------------
if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo ">> Creando key pair $KEY_NAME -> infra/${KEY_NAME}.pem"
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
    --query 'KeyMaterial' --output text > "infra/${KEY_NAME}.pem"
  chmod 600 "infra/${KEY_NAME}.pem"
fi

# --- Security Group --------------------------------------------------------
if ! aws ec2 describe-security-groups --region "$REGION" --group-names "$SG_NAME" >/dev/null 2>&1; then
  SG_ID="$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" \
    --description 'SCIF MLflow + SSH' --query 'GroupId' --output text)"
  # >>> CAPTURA CLAVE #1: este es el paso "habilitar el puerto en el Security Group"
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
    --ip-permissions \
      "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=ssh}]" \
      "IpProtocol=tcp,FromPort=5000,ToPort=5000,IpRanges=[{CidrIp=${MY_IP}/32,Description=mlflow-ui}]"
else
  SG_ID="$(aws ec2 describe-security-groups --region "$REGION" --group-names "$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)"
fi
echo ">> Security Group: $SG_ID  (SSH 22 + MLflow 5000 abiertos a ${MY_IP}/32)"

# --- Lanzar ---------------------------------------------------------------
RUN_JSON="$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true}" \
  --instance-initiated-shutdown-behavior stop \
  --user-data file://infra/mlflow_user_data.sh \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=${NAME_TAG}},{Key=project,Value=${TAG}}]" \
  --count 1)"

IID="$(echo "$RUN_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["Instances"][0]["InstanceId"])')"
echo ">> Instancia lanzada: $IID  (esperando IP publica...)"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
PUB_IP="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat > infra/ec2_instance.env <<EOF
REGION=$REGION
INSTANCE_ID=$IID
PUBLIC_IP=$PUB_IP
INSTANCE_TYPE=$INSTANCE_TYPE
KEY_PEM=infra/${KEY_NAME}.pem
SSH_USER=ubuntu
MLFLOW_TRACKING_URI=http://$PUB_IP:5000
EOF

echo "==========================================================="
echo " InstanceId : $IID"
echo " IP publica : $PUB_IP"
echo " SSH        : ssh -i infra/${KEY_NAME}.pem ubuntu@$PUB_IP"
echo " MLflow UI  : http://$PUB_IP:5000   (tarda ~3-5 min en subir)"
echo " Datos en   : infra/ec2_instance.env"
echo "==========================================================="
echo " >>> CAPTURA CLAVE #2: consola EC2 mostrando Instance ID + IP publica + usuario."
echo " >>> CAPTURA CLAVE #3: navegador en http://$PUB_IP:5000 con la IP visible en la barra."

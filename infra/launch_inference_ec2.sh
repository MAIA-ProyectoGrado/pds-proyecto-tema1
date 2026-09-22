#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Lanza UNA instancia EC2 pequena, solo para servir el prototipo de inferencia
# (api + dashboard con docker compose, ver ../docker-compose.yml). No instala
# MLflow ni sirve para entrenar: para eso esta launch_ec2.sh.
#
# Mismas restricciones del AWS Academy Learner Lab que launch_ec2.sh:
#   * ec2:RunInstances SOLO permite tamanos <= *.large.
#   * Instancias GPU: DENEGADAS.
#   * Key pair preexistente `vockey` (archivo local infra/vockey.pem) y
#     perfil de instancia `LabInstanceProfile` (credenciales S3 para dvc pull
#     dentro de la EC2, sin copiar credenciales de AWS a la maquina).
#
#   Region : us-east-1
#   Tipo   : t3.medium (2 vCPU / 4 GB, ~USD 0.0416/h) - alcanza para servir
#            SciBERT (pesos ~420 MB) a baja concurrencia. Si se queda corto
#            de memoria: INSTANCE_TYPE=t3.large ./infra/launch_inference_ec2.sh
#
# Uso:
#   ./infra/set_aws_credentials.sh        # si aun no hay credenciales validas
#   ./infra/launch_inference_ec2.sh
#   INSTANCE_TYPE=t3.large ./infra/launch_inference_ec2.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Git Bash en Windows reescribe rutas tipo /dev/sda1 como si fueran rutas de
# archivo locales; esto lo desactiva para los argumentos de la AWS CLI.
export MSYS_NO_PATHCONV=1

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
# Key pair propio (no el 'vockey' del lab): en el Learner Lab ese key pair se
# regenera en cada sesion nueva y no conservamos su clave privada. Creamos y
# reutilizamos uno propio para no depender de eso.
KEY_NAME="${KEY_NAME:-scif-inference-key}"
KEY_PEM="${KEY_PEM:-infra/scif-inference-key.pem}"
SG_NAME="${SG_NAME:-scif-inference-sg}"
IAM_PROFILE="${IAM_PROFILE:-LabInstanceProfile}"
VOLUME_GB="${VOLUME_GB:-30}"
NAME_TAG="${NAME_TAG:-scif-inference}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
ENV_FILE="infra/ec2_inference.env"
export AWS_DEFAULT_REGION="$REGION"

case "$INSTANCE_TYPE" in
  *.large|*.medium|*.small|*.micro|*.nano) : ;;
  *) echo "ERROR: '$INSTANCE_TYPE' > *.large -> el Learner Lab lo deniega."; exit 1 ;;
esac

echo ">> Verificando credenciales..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: credenciales de AWS invalidas o expiradas."
  echo "       Corre ./infra/set_aws_credentials.sh y vuelve a intentar."
  exit 1
fi

echo ">> Region=$REGION  Tipo=$INSTANCE_TYPE  Key=$KEY_NAME  Perfil=$IAM_PROFILE"
MY_IP="$(curl -s https://checkip.amazonaws.com || echo 0.0.0.0)"
echo ">> Tu IP publica: $MY_IP"

# --- AMI: Ubuntu 22.04 LTS amd64 (owner Canonical 099720109477) -------------
AMI_ID="${AMI_ID:-$(aws ec2 describe-images --owners 099720109477 \
  --filters 'Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*' \
            'Name=state,Values=available' 'Name=architecture,Values=x86_64' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)}"
echo ">> AMI: $AMI_ID"

# --- key pair ---------------------------------------------------------
# Si el key pair ya existe en AWS pero no tenemos el .pem local (por ejemplo,
# una sesion nueva del Learner Lab regenero 'vockey'), no hay forma de
# recuperar la clave privada: hay que recrear el par con otro nombre.
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  if [[ ! -f "$KEY_PEM" ]]; then
    echo "ERROR: el key pair '$KEY_NAME' ya existe en AWS pero no tengo $KEY_PEM local."
    echo "       Bórralo (aws ec2 delete-key-pair --key-name $KEY_NAME) o usa KEY_NAME=otro-nombre."
    exit 1
  fi
else
  echo ">> Creando key pair '$KEY_NAME'..."
  aws ec2 create-key-pair --key-name "$KEY_NAME" --key-type rsa \
    --query 'KeyMaterial' --output text > "$KEY_PEM"
fi
chmod 600 "$KEY_PEM"

# --- Security Group (habilitar 22 y el puerto del tablero solo a tu IP) ---
if ! aws ec2 describe-security-groups --group-names "$SG_NAME" >/dev/null 2>&1; then
  VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  SG_ID="$(aws ec2 create-security-group --group-name "$SG_NAME" \
    --description 'SCIF inference API + dashboard' --vpc-id "$VPC_ID" --query 'GroupId' --output text)"
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=ssh}]" \
    "IpProtocol=tcp,FromPort=${DASHBOARD_PORT},ToPort=${DASHBOARD_PORT},IpRanges=[{CidrIp=${MY_IP}/32,Description=dashboard}]"
else
  SG_ID="$(aws ec2 describe-security-groups --group-names "$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)"
  # Si cambio tu IP desde la ultima vez, asegura que siga autorizada.
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=ssh}]" \
    "IpProtocol=tcp,FromPort=${DASHBOARD_PORT},ToPort=${DASHBOARD_PORT},IpRanges=[{CidrIp=${MY_IP}/32,Description=dashboard}]" \
    2>/dev/null || true
fi
echo ">> Security Group: $SG_ID  (22 + ${DASHBOARD_PORT} abiertos a ${MY_IP}/32)"

# --- lanzar -------------------------------------------------------------
IID="$(aws ec2 run-instances \
  --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=${IAM_PROFILE}" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true}" \
  --instance-initiated-shutdown-behavior stop \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME_TAG}},{Key=project,Value=scif-inference}]" \
  --count 1 --query 'Instances[0].InstanceId' --output text)"

echo ">> Instancia: $IID  (esperando estado running...)"
aws ec2 wait instance-running --instance-ids "$IID"
PUB_IP="$(aws ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat > "$ENV_FILE" <<EOF
REGION=$REGION
INSTANCE_ID=$IID
PUBLIC_IP=$PUB_IP
INSTANCE_TYPE=$INSTANCE_TYPE
KEY_PEM=$KEY_PEM
SSH_USER=ubuntu
DASHBOARD_PORT=$DASHBOARD_PORT
DASHBOARD_URL=http://$PUB_IP:$DASHBOARD_PORT
EOF

echo "==========================================================="
echo " InstanceId : $IID"
echo " IP publica : $PUB_IP"
echo " SSH        : ssh -i $KEY_PEM ubuntu@$PUB_IP"
echo " Tablero    : http://$PUB_IP:$DASHBOARD_PORT   (tras correr deploy_ec2.sh)"
echo "==========================================================="

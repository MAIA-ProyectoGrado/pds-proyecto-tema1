#!/usr/bin/env bash
# Crea la infraestructura de SCIF en AWS y lanza la instancia que sirve la aplicación.
#
# Se ejecuta en el equipo local, con la AWS CLI conectada a la cuenta (`aws login`).
# Es idempotente: si un recurso ya existe, lo reutiliza.
#
#   1. Rol IAM `scif-ec2-dvc-read` con permiso de solo lectura sobre el bucket de DVC
#      (el de .dvc/config) y su perfil de instancia.
#   2. Par de llaves SSH `scif-key`; la llave privada queda en infra/scif-key.pem.
#   3. Grupo de seguridad `scif-sg`: puerto 22 solo para la IP actual y 8080 abierto.
#   4. Instancia Ubuntu 24.04 con disco gp3 cifrado, IMDSv2 obligatorio y el rol anterior.
#      El límite de saltos de IMDSv2 es 2 para que el contenedor de la API lea las
#      credenciales del rol.
#   5. IP elástica asociada a la instancia, para que el enlace no cambie.
#   6. Permiso del rol para consultar Nova Pro y Ministral 3 8B en Bedrock
#      (infra/enable_bedrock.sh).
#
# Después, en la instancia, se copia el repositorio y se ejecuta infra/deploy_ec2.sh.
#
# Variables opcionales:
#   REGION=us-east-2               región (por defecto, la de `aws configure`)
#   INSTANCE_TYPE=c7i-flex.large   tipo de instancia (2 vCPU, 4 GB)
#   VOLUME_GB=30                   tamaño del disco
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REGION="${REGION:-$(aws configure get region || echo us-east-2)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i-flex.large}"
VOLUME_GB="${VOLUME_GB:-30}"
ROLE="scif-ec2-dvc-read"
KEY_NAME="scif-key"
KEY_PEM="infra/scif-key.pem"
SG_NAME="scif-sg"
NAME_TAG="scif-app"
ENV_FILE="infra/ec2_inference.env"
export AWS_DEFAULT_REGION="$REGION"

aws sts get-caller-identity >/dev/null || { echo "!! Sin credenciales: ejecuta 'aws login'." >&2; exit 1; }
BUCKET="$(sed -n 's#.*url = s3://\([^/]*\).*#\1#p' .dvc/config | head -1)"
[ -n "$BUCKET" ] || { echo "!! No encontré el bucket de DVC en .dvc/config." >&2; exit 1; }
echo ">> Región $REGION, tipo $INSTANCE_TYPE, bucket $BUCKET"

# ── 1. Rol IAM de solo lectura sobre el bucket ──────────────────────────────
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  echo ">> Creando el rol $ROLE..."
  aws iam create-role --role-name "$ROLE" --description "SCIF: la instancia lee los modelos del bucket DVC" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --tags Key=project,Value=scif >/dev/null
  aws iam create-instance-profile --instance-profile-name "$ROLE" --tags Key=project,Value=scif >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  sleep 10   # IAM tarda unos segundos en propagar el perfil
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name scif-s3-read --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"s3:ListBucket\", \"s3:GetBucketLocation\"], \"Resource\": \"arn:aws:s3:::$BUCKET\"},
    {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\"], \"Resource\": \"arn:aws:s3:::$BUCKET/*\"}
  ]}"

# ── 2. Par de llaves SSH ────────────────────────────────────────────────────
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo ">> Creando el par de llaves $KEY_NAME..."
  aws ec2 create-key-pair --key-name "$KEY_NAME" --key-type ed25519 \
    --tag-specifications 'ResourceType=key-pair,Tags=[{Key=project,Value=scif}]' \
    --query KeyMaterial --output text > "$KEY_PEM"
  chmod 400 "$KEY_PEM"
elif [ ! -f "$KEY_PEM" ]; then
  echo "!! El par $KEY_NAME existe en AWS pero falta $KEY_PEM. Pide la llave a quien la creó." >&2
  exit 1
fi

# ── 3. Grupo de seguridad ───────────────────────────────────────────────────
MY_IP="$(curl -s https://checkip.amazonaws.com)"
SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  echo ">> Creando el grupo de seguridad $SG_NAME..."
  VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  SG_ID="$(aws ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC_ID" \
    --description "SCIF: tablero publico en 8080, SSH solo desde la IP del administrador" \
    --tag-specifications 'ResourceType=security-group,Tags=[{Key=project,Value=scif}]' \
    --query GroupId --output text)"
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions \
    "IpProtocol=tcp,FromPort=8080,ToPort=8080,IpRanges=[{CidrIp=0.0.0.0/0,Description=tablero}]" >/dev/null
fi
# SSH solo desde la IP actual; si cambió, se agrega la nueva.
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions \
  "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=ssh-administrador}]" >/dev/null 2>&1 || true
echo ">> Grupo de seguridad $SG_ID: 8080 abierto, 22 para ${MY_IP}/32"

# ── 4. Instancia ────────────────────────────────────────────────────────────
AMI_ID="$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)"
echo ">> Lanzando $INSTANCE_TYPE con Ubuntu 24.04 ($AMI_ID)..."
IID="$(aws ec2 run-instances --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" --iam-instance-profile "Name=$ROLE" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true,Encrypted=true}" \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=2 \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME_TAG},{Key=project,Value=scif}]" \
                       "ResourceType=volume,Tags=[{Key=project,Value=scif}]" \
  --count 1 --query 'Instances[0].InstanceId' --output text)"
aws ec2 wait instance-running --instance-ids "$IID"

# ── 5. IP elástica ──────────────────────────────────────────────────────────
ALLOC="$(aws ec2 allocate-address --domain vpc \
  --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME_TAG},{Key=project,Value=scif}]" \
  --query AllocationId --output text)"
aws ec2 associate-address --instance-id "$IID" --allocation-id "$ALLOC" >/dev/null
PUB_IP="$(aws ec2 describe-addresses --allocation-ids "$ALLOC" --query 'Addresses[0].PublicIp' --output text)"

cat > "$ENV_FILE" <<EOF
REGION=$REGION
INSTANCE_ID=$IID
ALLOCATION_ID=$ALLOC
PUBLIC_IP=$PUB_IP
KEY_PEM=$KEY_PEM
SSH_USER=ubuntu
EOF

# ── 6. Acceso a Bedrock ─────────────────────────────────────────────────────
REGION="$REGION" bash infra/enable_bedrock.sh

echo "==================================================================="
echo " Instancia : $IID ($INSTANCE_TYPE)"
echo " IP fija   : $PUB_IP"
echo " SSH       : ssh -i $KEY_PEM ubuntu@$PUB_IP"
echo " Siguiente : copiar el repositorio a la instancia y ejecutar infra/deploy_ec2.sh"
echo "==================================================================="

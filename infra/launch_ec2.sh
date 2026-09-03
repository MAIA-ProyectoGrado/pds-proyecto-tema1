#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Lanza UNA instancia EC2 (servidor MLflow + entrenamiento CPU) en AWS Academy
# Learner Lab. Restricciones detectadas del lab (politica Pvoclabs2):
#   * ec2:RunInstances SOLO permite tamanos  <= *.large  (2 vCPU).
#   * Instancias GPU (g4dn, g5, p3...) : DENEGADAS explicitamente.
#   * ssm:GetParameters de alias de AMI publicos : DENEGADO -> AMI se resuelve
#     con ec2:DescribeImages.
#   * Usar el key pair preexistente `vockey` (archivo local: llave.pem) y el
#     perfil de instancia `LabInstanceProfile` (da credenciales S3 a la EC2).
#
#   Region : us-east-1
#   Tipo   : t3.large  (2 vCPU / 8 GB, ~USD 0.083/h)  [c5.large tambien sirve]
#
# Uso:
#   ./infra/launch_ec2.sh
#   INSTANCE_TYPE=c5.large KEY_NAME=vockey ./infra/launch_ec2.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.large}"       # <= *.large obligatorio en el lab
KEY_NAME="${KEY_NAME:-vockey}"                   # key pair preexistente del lab
KEY_PEM="${KEY_PEM:-llave.pem}"                  # archivo local de esa llave
SG_NAME="${SG_NAME:-scif-mlflow-sg}"
IAM_PROFILE="${IAM_PROFILE:-LabInstanceProfile}"
VOLUME_GB="${VOLUME_GB:-60}"
NAME_TAG="${NAME_TAG:-scif-mlflow-train}"
export AWS_DEFAULT_REGION="$REGION"

case "$INSTANCE_TYPE" in
  *.large|*.medium|*.small|*.micro|*.nano) : ;;
  *) echo "ERROR: '$INSTANCE_TYPE' > *.large -> el Learner Lab lo deniega."; exit 1 ;;
esac

echo ">> Region=$REGION  Tipo=$INSTANCE_TYPE  Key=$KEY_NAME  Perfil=$IAM_PROFILE"
MY_IP="$(curl -s https://checkip.amazonaws.com || echo 0.0.0.0)"
echo ">> Tu IP publica: $MY_IP"

# --- AMI: Ubuntu 22.04 LTS amd64 (owner Canonical 099720109477) -------------
AMI_ID="${AMI_ID:-$(aws ec2 describe-images --owners 099720109477 \
  --filters 'Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*' \
            'Name=state,Values=available' 'Name=architecture,Values=x86_64' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)}"
echo ">> AMI: $AMI_ID"

# --- key pair -------------------------------------------------------------
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo "ERROR: el key pair '$KEY_NAME' no existe en el lab. Usa 'vockey'."; exit 1
fi
[[ -f "$KEY_PEM" ]] && chmod 600 "$KEY_PEM" 2>/dev/null || \
  echo ">> AVISO: no encuentro $KEY_PEM localmente; lo necesitaras para el ssh."

# --- Security Group (habilitar 22 y 5000 solo a tu IP) --------------------
if ! aws ec2 describe-security-groups --group-names "$SG_NAME" >/dev/null 2>&1; then
  VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  SG_ID="$(aws ec2 create-security-group --group-name "$SG_NAME" \
    --description 'SCIF MLflow + SSH' --vpc-id "$VPC_ID" --query 'GroupId' --output text)"
  # >>> CAPTURA CLAVE #1: "habilitar el puerto en las reglas de entrada del Security Group"
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=ssh}]" \
    "IpProtocol=tcp,FromPort=5000,ToPort=5000,IpRanges=[{CidrIp=${MY_IP}/32,Description=mlflow-ui}]"
else
  SG_ID="$(aws ec2 describe-security-groups --group-names "$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)"
fi
echo ">> Security Group: $SG_ID  (22 + 5000 abiertos a ${MY_IP}/32)"

# --- lanzar -------------------------------------------------------------
IID="$(aws ec2 run-instances \
  --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=${IAM_PROFILE}" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true}" \
  --instance-initiated-shutdown-behavior stop \
  --user-data file://infra/mlflow_user_data.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME_TAG}},{Key=project,Value=scif-entrega2}]" \
  --count 1 --query 'Instances[0].InstanceId' --output text)"

echo ">> Instancia: $IID  (esperando estado running...)"
aws ec2 wait instance-running --instance-ids "$IID"
PUB_IP="$(aws ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat > infra/ec2_instance.env <<EOF
REGION=$REGION
INSTANCE_ID=$IID
PUBLIC_IP=$PUB_IP
INSTANCE_TYPE=$INSTANCE_TYPE
KEY_PEM=$KEY_PEM
SSH_USER=ubuntu
MLFLOW_TRACKING_URI=http://$PUB_IP:5000
EOF

echo "==========================================================="
echo " InstanceId : $IID"
echo " IP publica : $PUB_IP"
echo " SSH        : ssh -i $KEY_PEM ubuntu@$PUB_IP"
echo " MLflow UI  : http://$PUB_IP:5000   (tarda ~3-5 min)"
echo "==========================================================="
echo " >>> CAPTURA CLAVE #2: consola EC2 con Instance ID + IP publica + usuario."
echo " >>> CAPTURA CLAVE #3: navegador en http://$PUB_IP:5000 con la IP visible."

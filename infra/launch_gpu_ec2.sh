#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Aprovisiona UNA instancia EC2 con GPU para el entrenamiento v3 (cuenta propia
# 779319895642). Presupuesto de referencia: USD 100 y 20 h.
#
#   Region : us-east-1
#   Tipo   : g5.xlarge  (1x A10G 24 GB)  ~USD 1.006/h  [g4dn.xlarge ~USD 0.526/h]
#
# PRERREQUISITO: la cuenta AWS debe estar en un **plan de pago** (no "Free Plan")
# y con cuota > 0 de "Running On-Demand G and VT instances". Si `run-instances`
# responde "not eligible for Free Tier", falta actualizar el plan en la consola
# de Billing. La cuota se pide con:
#   aws service-quotas request-service-quota-increase --service-code ec2 \
#     --quota-code L-DB2E81BA --desired-value 8
#
# Uso:
#   source scripts/awsenv.sh          # exporta AWS_ACCESS_KEY_ID / SECRET desde .env
#   ./infra/launch_gpu_ec2.sh
#   INSTANCE_TYPE=g4dn.xlarge ./infra/launch_gpu_ec2.sh
# ---------------------------------------------------------------------------
set -euo pipefail
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"

INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
KEY_NAME="${KEY_NAME:-scif-v3}"
KEY_PEM="${KEY_PEM:-infra/scif-v3.pem}"
SG_NAME="${SG_NAME:-scif-v3-sg}"
IAM_PROFILE="${IAM_PROFILE:-scif-ec2-s3}"
BUCKET="${BUCKET:-scif-dvcstore-779319895642-use1}"
VOLUME_GB="${VOLUME_GB:-120}"

MYIP="$(curl -s https://checkip.amazonaws.com)"
echo ">> IP: $MYIP  Tipo: $INSTANCE_TYPE"

# AMI: Deep Learning OSS Nvidia Driver (Ubuntu 22.04) - trae CUDA + drivers
AMI_ID="$(aws ssm get-parameters \
  --names /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.4-ubuntu-22.04/latest/ami-id \
  --query 'Parameters[0].Value' --output text 2>/dev/null || true)"
if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
  AMI_ID="$(aws ec2 describe-images --owners amazon \
    --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*' 'Name=state,Values=available' \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)"
fi
echo ">> AMI: $AMI_ID"

# Security group
if ! aws ec2 describe-security-groups --group-names "$SG_NAME" >/dev/null 2>&1; then
  VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  SG="$(aws ec2 create-security-group --group-name "$SG_NAME" --description 'SCIF v3' --vpc-id "$VPC" --query GroupId --output text)"
  aws ec2 authorize-security-group-ingress --group-id "$SG" --ip-permissions \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MYIP}/32}]" \
    "IpProtocol=tcp,FromPort=5000,ToPort=5000,IpRanges=[{CidrIp=${MYIP}/32}]"
else
  SG="$(aws ec2 describe-security-groups --group-names "$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)"
fi
echo ">> SG: $SG"

USERDATA="$(cat <<EOF
#!/bin/bash
set -eux
exec > /var/log/scif-userdata.log 2>&1
apt-get update -y && apt-get install -y python3-venv git awscli
mkdir -p /opt/mlflow/artifacts && chown -R ubuntu:ubuntu /opt/mlflow
sudo -u ubuntu python3 -m venv /opt/mlflow/venv
sudo -u ubuntu /opt/mlflow/venv/bin/pip -q install "mlflow==2.17.2" boto3
cat >/etc/systemd/system/mlflow.service <<UNIT
[Unit]
Description=MLflow
After=network.target
[Service]
User=ubuntu
WorkingDirectory=/opt/mlflow
ExecStart=/opt/mlflow/venv/bin/mlflow server --backend-store-uri sqlite:////opt/mlflow/mlflow.db --default-artifact-root s3://${BUCKET}/mlflow-artifacts --host 0.0.0.0 --port 5000
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable --now mlflow.service
echo '0 * * * * root bash -c "UP=\$(awk \"{print int(\\\$1/3600)}\" /proc/uptime); [ \"\\\$UP\" -ge 19 ] && /sbin/shutdown -h now"' >/etc/cron.d/auto-stop
touch /home/ubuntu/USERDATA_DONE
EOF
)"

IID="$(aws ec2 run-instances --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG" --iam-instance-profile "Name=${IAM_PROFILE}" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3}" \
  --instance-initiated-shutdown-behavior stop \
  --user-data "$USERDATA" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=scif-v3-gpu}]' \
  --count 1 --query 'Instances[0].InstanceId' --output text)"
aws ec2 wait instance-running --instance-ids "$IID"
IP="$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat > infra/ec2_v3.env <<EOF
INSTANCE_ID=$IID
PUBLIC_IP=$IP
KEY_PEM=$KEY_PEM
MLFLOW_TRACKING_URI=http://$IP:5000
EOF
echo "==========================================================="
echo " InstanceId : $IID       IP: $IP"
echo " SSH        : ssh -i $KEY_PEM ubuntu@$IP"
echo " MLflow     : http://$IP:5000   (~3 min)"
echo " Siguiente  : scp infra/train_v3_gpu.sh + .env, luego bash train_v3_gpu.sh"
echo " CAPTURAS   : consola EC2 (ID+IP+usuario), SG puerto 5000, MLflow UI con IP"
echo "==========================================================="

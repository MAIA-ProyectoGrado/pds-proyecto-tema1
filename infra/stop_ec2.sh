#!/usr/bin/env bash
# Detiene (NO termina) la instancia de la Entrega 2. Requisito del enunciado:
# "Mantenga su maquina en EC2 con MLflow detenida (no la termine)."
set -euo pipefail
source infra/ec2_instance.env
echo ">> Deteniendo $INSTANCE_ID en $REGION ..."
aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$INSTANCE_ID"
echo ">> Estado:"
aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text
echo ">> Instancia DETENIDA (no terminada). El volumen EBS sigue facturando ~USD 0.08/GB-mes."
echo ">> Para reanudar: aws ec2 start-instances --region $REGION --instance-ids $INSTANCE_ID"

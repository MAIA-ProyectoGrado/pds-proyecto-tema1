#!/usr/bin/env bash
# Da a la instancia de SCIF acceso a Nova Pro y Ministral 3 8B en Amazon Bedrock.
#
# Se ejecuta en el equipo local, con la AWS CLI conectada a la cuenta (`aws login`).
# Es idempotente y lo llama también launch_inference_ec2.sh.
#
#   1. Agrega al rol `scif-ec2-dvc-read` la política `scif-bedrock-invoke`, que solo
#      permite bedrock:InvokeModel sobre esos dos modelos (y sobre el perfil de
#      inferencia de EE. UU. por el que se consulta Nova Pro).
#   2. Si infra/ec2_inference.env tiene INSTANCE_ID, sube a 2 el límite de saltos
#      del servicio de metadatos (IMDSv2) de la instancia. Con 1, el contenedor de
#      la API no alcanza las credenciales del rol.
#
# Variables opcionales:
#   REGION=us-east-2    región de Bedrock y de la instancia
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_FILE="infra/ec2_inference.env"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
REGION="${REGION:-us-east-2}"
ROLE="scif-ec2-dvc-read"
export AWS_DEFAULT_REGION="$REGION"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)" \
  || { echo "!! Sin credenciales: ejecuta 'aws login'." >&2; exit 1; }

# Nova Pro se consulta por el perfil de inferencia us.amazon.nova-pro-v1:0, que puede
# atender la petición en us-east-1, us-east-2 o us-west-2: el permiso cubre el perfil y
# el modelo base en esas tres regiones.
echo ">> Política scif-bedrock-invoke en el rol $ROLE (región $REGION)..."
aws iam put-role-policy --role-name "$ROLE" --policy-name scif-bedrock-invoke --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Sid\": \"NovaProPerfilEEUU\", \"Effect\": \"Allow\", \"Action\": \"bedrock:InvokeModel\",
     \"Resource\": [
       \"arn:aws:bedrock:$REGION:$ACCOUNT:inference-profile/us.amazon.nova-pro-v1:0\",
       \"arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0\",
       \"arn:aws:bedrock:us-east-2::foundation-model/amazon.nova-pro-v1:0\",
       \"arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-pro-v1:0\"]},
    {\"Sid\": \"Ministral3\", \"Effect\": \"Allow\", \"Action\": \"bedrock:InvokeModel\",
     \"Resource\": \"arn:aws:bedrock:$REGION::foundation-model/mistral.ministral-3-8b-instruct\"}
  ]}"

if [ -n "${INSTANCE_ID:-}" ]; then
  echo ">> Límite de saltos de IMDSv2 en 2 para $INSTANCE_ID..."
  aws ec2 modify-instance-metadata-options --instance-id "$INSTANCE_ID" \
    --http-tokens required --http-endpoint enabled --http-put-response-hop-limit 2 >/dev/null
else
  echo ">> Sin INSTANCE_ID en $ENV_FILE: no se cambió ninguna instancia."
fi

echo ">> Listo. En la instancia, despliega con SCIF_LLM_ENABLED=true (infra/deploy_ec2.sh lo hace por defecto)."

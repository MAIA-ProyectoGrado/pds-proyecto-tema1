#!/usr/bin/env bash
# Despliega SCIF (API + tablero) en una instancia EC2 con Ubuntu y Docker.
#
# Se ejecuta DENTRO de la instancia, desde la raíz del repositorio:
#
#   bash infra/deploy_ec2.sh
#
# Pasos:
#   1. Instala Docker y DVC si faltan (paquetes de Ubuntu y PyPI).
#   2. Descarga los pesos de SciBERT y MiniLM con `dvc pull`. Las credenciales
#      salen del rol IAM de la instancia: en la máquina no hay llaves de AWS.
#   3. Guarda en .env si los modelos de lenguaje quedan activos, construye las
#      imágenes en la propia instancia (x86_64), levanta los servicios y espera a
#      que la API quede sana.
#   4. Comprueba que el contenedor de la API obtenga las credenciales del rol para
#      consultar Bedrock.
#
# Si los pesos no quedan en disco, el script se detiene con un error: el
# despliegue público siempre sirve el modelo recomendado.
#
# Variables opcionales:
#   SCIF_PORT=8080            puerto público del tablero
#   SKIP_DVC=1                no ejecutar dvc pull (pesos copiados a mano en models/)
#   SCIF_LLM_ENABLED=true     ofrecer Nova Pro y Ministral 3 8B (Amazon Bedrock); el rol
#                             necesita el permiso de infra/enable_bedrock.sh. false los oculta.
#                             El valor queda en .env, así que un `docker compose up`
#                             posterior lo conserva.
#   Los demás parámetros (tope diario, consultas por minuto…) se leen de .env.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DVC_VENV="${HOME}/.venv-dvc"
DVC_VERSION="3.67.1"
PORT="${SCIF_PORT:-8080}"
LLM="${SCIF_LLM_ENABLED:-true}"
WEIGHTS=(models/scif-scibert-chunks models/minilm-l6-v2)

echo ">> Proyecto: $REPO_DIR"

# ── 1. Docker y DVC ─────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo ">> Instalando Docker (paquetes de Ubuntu)..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker.io docker-compose-v2
  sudo usermod -aG docker "$USER"
fi
# Recién instalado, el grupo docker aplica en la próxima sesión; mientras tanto, sudo.
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

if [ -z "${SKIP_DVC:-}" ] && [ ! -x "$DVC_VENV/bin/dvc" ]; then
  echo ">> Instalando DVC $DVC_VERSION con soporte S3..."
  sudo apt-get install -y -qq python3-venv
  python3 -m venv "$DVC_VENV"
  "$DVC_VENV/bin/pip" install -q "dvc[s3]==$DVC_VERSION"
fi

# ── 2. Pesos de los modelos ─────────────────────────────────────────────────
if [ -z "${SKIP_DVC:-}" ]; then
  echo ">> Descargando pesos con DVC (rol IAM de la instancia)..."
  "$DVC_VENV/bin/dvc" pull "${WEIGHTS[@]/%/.dvc}"
fi
for w in "${WEIGHTS[@]}"; do
  if [ ! -f "$w/config.json" ]; then
    echo "!! Faltan los pesos en $w. Revisa el rol IAM de la instancia o copia la carpeta a mano." >&2
    exit 1
  fi
done
echo ">> Pesos listos: ${WEIGHTS[*]}"

# ── 3. Construir y levantar ─────────────────────────────────────────────────
case "$LLM" in
  true|false) ;;
  *) echo "!! SCIF_LLM_ENABLED debe ser true o false (recibido: $LLM)." >&2; exit 1 ;;
esac
# .env conserva el resto de los parámetros; solo se reemplaza esta línea.
touch .env
{ grep -v '^SCIF_LLM_ENABLED=' .env || true; echo "SCIF_LLM_ENABLED=$LLM"; } > .env.tmp
mv .env.tmp .env
echo ">> Modelos de lenguaje: SCIF_LLM_ENABLED=$LLM (guardado en .env)"

echo ">> Construyendo y levantando los servicios (la primera vez tarda varios minutos)..."
SCIF_PORT="$PORT" SCIF_LLM_ENABLED="$LLM" $DOCKER compose up -d --build

echo ">> Esperando a que la API quede sana..."
estado="unknown"
for _ in $(seq 1 60); do
  estado="$($DOCKER inspect -f '{{.State.Health.Status}}' scif-api 2>/dev/null || echo unknown)"
  [ "$estado" = "healthy" ] && break
  sleep 5
done
if [ "$estado" != "healthy" ]; then
  echo "!! La API no quedó sana. Revisa: $DOCKER compose logs api" >&2
  exit 1
fi

# ── 4. Acceso a Bedrock desde el contenedor (llamada gratuita a STS) ────────
if [ "$LLM" = "true" ]; then
  if $DOCKER exec scif-api python -c "import boto3; boto3.client('sts').get_caller_identity()" >/dev/null 2>&1; then
    echo ">> El contenedor de la API tiene las credenciales del rol: Nova Pro y Ministral 3 8B quedan activos."
  else
    echo "!! El contenedor no obtuvo credenciales del rol. Desde el equipo local, ejecuta" >&2
    echo "!! infra/enable_bedrock.sh (límite de saltos de IMDSv2 y permiso de Bedrock)." >&2
  fi
fi

TOKEN="$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)"
IP="$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 || true)"

echo
$DOCKER compose ps
echo
echo "==================================================================="
echo " Tablero:       http://${IP:-<IP-de-la-instancia>}:${PORT}"
echo " Estado:        http://${IP:-<IP-de-la-instancia>}:${PORT}/api/health"
echo " Documentación: http://${IP:-<IP-de-la-instancia>}:${PORT}/api/docs"
echo "==================================================================="
echo
echo "Comandos útiles:"
echo "  $DOCKER compose logs -f api     # registros del servicio"
echo "  $DOCKER compose restart         # reiniciar"
echo "  $DOCKER compose down            # detener los servicios"

#!/usr/bin/env bash
# Despliega el prototipo (API + tablero) en una EC2 con Docker.
#
# Se ejecuta DENTRO de la instancia, por ssh. La imagen se construye en la propia
# EC2: es x86_64 y las máquinas del equipo son arm64, así que construir local y
# copiar la imagen daría una arquitectura incompatible.
#
#   ssh -i llave.pem ubuntu@<IP>
#   git clone <repo> && cd pds-proyecto-tema1
#   bash infra/deploy_ec2.sh
#
# Variables opcionales:
#   SCIF_PORT=8080     puerto público del tablero
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo ">> Directorio del proyecto: $REPO_DIR"

# ── 1. Docker ───────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo ">> Instalando Docker..."
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo ">> Docker instalado. Cierra la sesión ssh y vuelve a entrar para usarlo sin sudo,"
  echo "   luego ejecuta de nuevo este script."
  exit 0
fi

docker compose version >/dev/null 2>&1 || {
  echo "!! Falta el plugin 'docker compose'. Instálalo y reintenta." >&2
  exit 1
}

# ── 2. Pesos de los modelos ─────────────────────────────────────────────────
# La línea base viaja en el repositorio; SciBERT se versiona con DVC.
if [ ! -f models/scif-scibert/config.json ]; then
  echo ">> Faltan los pesos de SciBERT en models/scif-scibert/"
  if command -v dvc >/dev/null 2>&1; then
    echo ">> Intentando 'dvc pull models/scif-scibert.dvc'..."
    dvc pull models/scif-scibert.dvc || echo "!! dvc pull falló (¿credenciales de S3?)."
  else
    echo "   Instala DVC ('pip install dvc[s3]') y ejecuta 'dvc pull models/scif-scibert.dvc',"
    echo "   o copia la carpeta con scp."
  fi
fi

if [ -f models/scif-scibert/config.json ]; then
  echo ">> SciBERT disponible: clasificación y recuperación de pasajes activas."
else
  echo ">> AVISO: sin SciBERT la API sirve solo la línea base y la recuperación"
  echo "   de pasajes queda deshabilitada. El tablero lo indica en la interfaz."
fi

# ── 3. Construir y levantar ─────────────────────────────────────────────────
echo ">> Construyendo imágenes (la primera vez descarga torch: varios minutos)..."
docker compose build

echo ">> Levantando servicios..."
docker compose up -d

# ── 4. Esperar a que el servicio quede sano ─────────────────────────────────
echo ">> Esperando a que la API responda..."
for _ in $(seq 1 60); do
  estado="$(docker inspect -f '{{.State.Health.Status}}' scif-api 2>/dev/null || echo unknown)"
  [ "$estado" = "healthy" ] && break
  sleep 5
done

PORT="${SCIF_PORT:-8080}"
IP_PUBLICA="$(curl -s --max-time 5 http://169.254.169.254/latest/meta-data/public-ipv4 || echo '<IP-de-la-instancia>')"

echo
docker compose ps
echo
echo "==================================================================="
echo " Tablero:  http://${IP_PUBLICA}:${PORT}"
echo " API:      http://${IP_PUBLICA}:${PORT}/api/models"
echo "==================================================================="
echo
echo "Abre el puerto ${PORT}/tcp en el grupo de seguridad de la instancia."
echo
echo "Comandos útiles:"
echo "  docker compose logs -f api     # registros del servicio"
echo "  docker compose restart         # reiniciar"
echo "  docker compose down            # detener (la entrega pide NO terminar la máquina)"

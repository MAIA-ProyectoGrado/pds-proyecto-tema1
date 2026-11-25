#!/usr/bin/env bash
# Configura credenciales de AWS (perfil "default") de forma interactiva, sin
# tener que editar ~/.aws/credentials a mano. Pensado para AWS Academy Learner
# Lab: sus credenciales son temporales (incluyen aws_session_token) y expiran
# cada pocas horas, así que este script se corre de nuevo cada vez que el lab
# entrega credenciales nuevas.
#
# Uso:
#   ./infra/set_aws_credentials.sh
#
# Acepta pegar el bloque completo tal cual lo entrega el panel "AWS Details"
# del Learner Lab:
#
#   [default]
#   aws_access_key_id=AKIA...
#   aws_secret_access_key=...
#   aws_session_token=...
#
# Termina el pegado con una línea vacía (Enter) o Ctrl+D. Si en vez de eso
# escribes un solo valor por línea, también pregunta uno por uno.
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo ">> Pega el bloque de credenciales de AWS Academy (Learner Lab > AWS Details > CLI)."
echo ">> Termina con una línea vacía. Si prefieres, Ctrl+D funciona también."
echo

PASTE=""
while IFS= read -r line; do
  [[ -z "$line" ]] && break
  PASTE+="$line"$'\n'
done

ACCESS_KEY="$(printf '%s' "$PASTE" | grep -im1 '^aws_access_key_id' | cut -d '=' -f2- | tr -d '[:space:]')"
SECRET_KEY="$(printf '%s' "$PASTE" | grep -im1 '^aws_secret_access_key' | cut -d '=' -f2- | tr -d '[:space:]')"
SESSION_TOKEN="$(printf '%s' "$PASTE" | grep -im1 '^aws_session_token' | cut -d '=' -f2- | tr -d '[:space:]')"

# Si no vino como bloque (por ejemplo se pegó vacío), pedir campo por campo.
if [[ -z "$ACCESS_KEY" ]]; then
  read -rp "AWS Access Key ID: " ACCESS_KEY
fi
if [[ -z "$SECRET_KEY" ]]; then
  read -rsp "AWS Secret Access Key: " SECRET_KEY
  echo
fi
if [[ -z "$SESSION_TOKEN" ]]; then
  read -rsp "AWS Session Token (Enter si no aplica, ej. cuenta propia sin Learner Lab): " SESSION_TOKEN
  echo
fi

if [[ -z "$ACCESS_KEY" || -z "$SECRET_KEY" ]]; then
  echo "!! Falta access key o secret key. Nada que configurar." >&2
  exit 1
fi

mkdir -p "$HOME/.aws"
CRED_FILE="$HOME/.aws/credentials"

# Reemplaza (o crea) solo la sección [default]; deja intactas otras secciones.
python3 - "$CRED_FILE" "$ACCESS_KEY" "$SECRET_KEY" "$SESSION_TOKEN" <<'PYEOF'
import configparser, sys, os

path, access_key, secret_key, session_token = sys.argv[1:5]
cfg = configparser.ConfigParser()
if os.path.exists(path):
    cfg.read(path)
if "default" not in cfg:
    cfg["default"] = {}
cfg["default"]["aws_access_key_id"] = access_key
cfg["default"]["aws_secret_access_key"] = secret_key
if session_token:
    cfg["default"]["aws_session_token"] = session_token
elif "aws_session_token" in cfg["default"]:
    del cfg["default"]["aws_session_token"]

with open(path, "w") as f:
    cfg.write(f)
os.chmod(path, 0o600)
PYEOF

aws configure set region "$REGION" --profile default
aws configure set output json --profile default

echo ">> Credenciales guardadas en $CRED_FILE (perfil default, region $REGION)."
echo ">> Verificando..."
aws sts get-caller-identity --output table

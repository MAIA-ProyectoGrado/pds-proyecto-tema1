#!/bin/bash
# Carga las credenciales AWS del .env (NO versionado) a variables de entorno.
#   source scripts/awsenv.sh
# .env esperado (ver .env.example):
#   AWS_ACESS_KEY=...
#   AWS_SECRET_ACCESS_KEY=...
ENV_FILE="${ENV_FILE:-$(dirname "${BASH_SOURCE[0]}")/../.env}"
[ -f "$ENV_FILE" ] || { echo "Falta $ENV_FILE (copia .env.example y rellena)"; return 1 2>/dev/null || exit 1; }
export AWS_ACCESS_KEY_ID="$(grep -E '^AWS_ACESS_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '\r"'"'"' ')"
export AWS_SECRET_ACCESS_KEY="$(grep -E '^AWS_SECRET_ACCESS_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '\r"'"'"' ')"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
unset AWS_SESSION_TOKEN AWS_PROFILE
echo "AWS listo: $(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '(credenciales invalidas)')"

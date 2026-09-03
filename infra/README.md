# infra/ — Servidor de experimentos (Entrega 2)

Una sola EC2 hace de **servidor MLflow** y de **máquina de entrenamiento**.
Región `us-east-1`. Presupuesto: ~USD 20, máx. 15 h de uso, **detener (no terminar)** al final.

| Script | Dónde se corre | Qué hace |
|---|---|---|
| `launch_ec2.sh` | local | crea key pair + Security Group (abre 22 y 5000 a tu IP) y lanza la instancia con `mlflow_user_data.sh`. Escribe `ec2_instance.env`. |
| `mlflow_user_data.sh` | (user-data, dentro de la EC2) | instala MLflow 3.15.2 como servicio systemd (`sqlite` + artefactos locales) en `:5000` + cron de auto-stop a las 14 h. |
| `remote_setup.sh` | dentro de la EC2 (ssh) | clona la rama, venv, `requirements-train.txt`, `dvc pull`, EDA, `train.py` v1 y v2, `analyze_metrics.py`. |
| `stop_ec2.sh` | local | `aws ec2 stop-instances` (no terminate) + verifica estado. |

## Uso rápido

```bash
# 1. lanzar (elige tipo segun cuota de GPU)
INSTANCE_TYPE=g4dn.xlarge ./infra/launch_ec2.sh        # GPU T4, ~USD 0.53/h
# INSTANCE_TYPE=c7i.2xlarge USE_GPU_AMI=0 ./infra/launch_ec2.sh   # fallback CPU

source infra/ec2_instance.env
curl -sf "$MLFLOW_TRACKING_URI/health" && echo OK       # esperar 3-5 min

# 2. pantallazos (docs/soportes/01_ec2_consola.png, 02_security_group.png, 03_mlflow_ui.png)

# 3. entrenar
ssh -i infra/scif-entrega2.pem ubuntu@$PUBLIC_IP
#   dentro: bash <(curl -sL .../infra/remote_setup.sh)

# 4. cerrar
./infra/stop_ec2.sh
```

## Costos (on-demand, us-east-1, aprox.)

| Tipo | GPU | USD/h | 15 h (techo) | uso real esperado (~5 h) |
|---|---|--:|--:|--:|
| g5.xlarge | A10G 24 GB | 1.006 | 15.1 | ~5 |
| g4dn.xlarge | T4 16 GB | 0.526 | 7.9 | ~2.6 |
| c7i.2xlarge | — (CPU) | 0.357 | 5.4 | ~1.8 |

+ EBS gp3 120 GB ≈ USD 0.08/GB-mes → ~USD 0.32 mientras exista el volumen.
Pon una alerta en **AWS Budgets** a USD 15.

## Guardarraíles de presupuesto

- `run-instances --instance-initiated-shutdown-behavior stop` → un `shutdown` no destruye nada.
- cron `auto-stop` a las 14 h de uptime (en `mlflow_user_data.sh`).
- `stop_ec2.sh` al terminar; el enunciado exige **stopped, no terminated**.
- Nunca lanzar más de 1 instancia a la vez.

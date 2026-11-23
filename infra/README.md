# infra/: despliegue en AWS

Dos scripts dejan la aplicación pública en una instancia EC2. Los pesos de los
modelos se descargan del bucket de DVC con el rol de la instancia, así que en la
máquina no hay llaves de AWS.

| Script | Dónde se corre | Qué hace |
|---|---|---|
| `launch_inference_ec2.sh` | Equipo local, con la AWS CLI conectada (`aws login` o `aws configure`) | Crea o reutiliza el rol IAM de solo lectura sobre el bucket de `.dvc/config`, el par de llaves `scif-key`, el grupo de seguridad `scif-sg` (8080 abierto, 22 solo para tu IP), la instancia Ubuntu 24.04 con IMDSv2 y disco cifrado, y una IP elástica. Escribe `infra/ec2_inference.env` (no se versiona). |
| `deploy_ec2.sh` | Dentro de la instancia, desde la raíz del repositorio | Instala Docker y DVC, descarga SciBERT y MiniLM con `dvc pull`, construye las imágenes, levanta la API y el tablero y espera a que la API quede sana. |

## Pasos

```bash
# 1. En el equipo local
REGION=us-east-2 ./infra/launch_inference_ec2.sh
source infra/ec2_inference.env

# 2. En la instancia
ssh -i "$KEY_PEM" "$SSH_USER@$PUBLIC_IP"
git clone https://github.com/MAIA-ProyectoGrado/pds-proyecto-tema1.git
cd pds-proyecto-tema1 && bash infra/deploy_ec2.sh
```

Para desplegar una copia local que aún no está en GitHub, el paso 2 puede
reemplazarse por un `rsync` del repositorio (con `.git`, que DVC necesita), sin
los entornos virtuales, los pesos ni `.dvc/cache/`.

El tablero queda en `http://<PUBLIC_IP>:8080` y la documentación de la API en
`http://<PUBLIC_IP>:8080/api/docs`.

Variables opcionales de `launch_inference_ec2.sh`: `REGION`, `INSTANCE_TYPE`
(por defecto `c7i-flex.large`) y `VOLUME_GB` (por defecto 30). De
`deploy_ec2.sh`: `SCIF_PORT` y `SKIP_DVC=1`, para cuando los pesos se copiaron a
mano en `models/`.

## Instancia

`c7i-flex.large`: 2 vCPU y 4 GB. Es la única de 4 GB que admite el plan
gratuito de AWS. La API ocupa unos 0,8 GB con los modelos cargados. Precio de
lista en `us-east-2`: USD 0,085 por hora, unos USD 2,2 al día con el disco y la
IP elástica.

## Apagar y borrar los recursos

```bash
source infra/ec2_inference.env
export AWS_DEFAULT_REGION=$REGION

# Detener la instancia (conserva el disco; la IP elástica sigue cobrándose)
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"

# Borrar todo
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
aws ec2 release-address --allocation-id "$ALLOCATION_ID"
aws ec2 delete-security-group --group-name scif-sg
aws ec2 delete-key-pair --key-name scif-key
aws iam remove-role-from-instance-profile --instance-profile-name scif-ec2-dvc-read --role-name scif-ec2-dvc-read
aws iam delete-instance-profile --instance-profile-name scif-ec2-dvc-read
aws iam delete-role-policy --role-name scif-ec2-dvc-read --policy-name scif-s3-read
aws iam delete-role --role-name scif-ec2-dvc-read
```

El bucket de DVC no se borra con estos comandos: guarda el dataset y los modelos.

## Historial

Los scripts de la entrega 2 (servidor de MLflow y entrenamiento en GPU en el
AWS Academy Learner Lab) están en [`historial/`](historial/README.md).

# infra/: despliegue en AWS

Tres scripts dejan la aplicación pública en una instancia EC2. Los pesos de los
modelos se descargan del bucket de DVC con el rol de la instancia, y el mismo rol
consulta Nova Pro y Ministral 3 8B en Bedrock, así que en la máquina no hay llaves
de AWS.

| Script | Dónde se corre | Qué hace |
|---|---|---|
| `launch_inference_ec2.sh` | Equipo local, con la AWS CLI conectada (`aws login` o `aws configure`) | Crea o reutiliza el rol IAM de solo lectura sobre el bucket de `.dvc/config`, el par de llaves `scif-key`, el grupo de seguridad `scif-sg` (8080 abierto, 22 solo para tu IP), la instancia Ubuntu 24.04 con IMDSv2 y disco cifrado, y una IP elástica. Escribe `infra/ec2_inference.env` (no se versiona) y al final llama a `enable_bedrock.sh`. |
| `enable_bedrock.sh` | Equipo local, con la AWS CLI conectada | Agrega al rol la política `scif-bedrock-invoke`, que solo permite `bedrock:InvokeModel` sobre Nova Pro y Ministral 3 8B, y sube a 2 el límite de saltos de IMDSv2 de la instancia para que el contenedor lea las credenciales del rol. Sirve para una instancia que ya existe. |
| `deploy_ec2.sh` | Dentro de la instancia, desde la raíz del repositorio | Instala Docker y DVC, descarga SciBERT y MiniLM con `dvc pull`, construye las imágenes, levanta la API y el tablero con los modelos de lenguaje activos, espera a que la API quede sana y comprueba que el contenedor tenga las credenciales del rol. |

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
`deploy_ec2.sh`: `SCIF_PORT`, `SKIP_DVC=1`, para cuando los pesos se copiaron a
mano en `models/`, y `SCIF_LLM_ENABLED=false`, para desplegar sin los modelos de
lenguaje.

## Instancia

`c7i-flex.large`: 2 vCPU y 4 GB. Es la única de 4 GB que admite el plan
gratuito de AWS. La API ocupa unos 0,8 GB con los modelos cargados.

## Costos

Precios de lista de `us-east-2`, consultados en la API de precios de AWS el 24 de
noviembre de 2026. No incluyen créditos ni el plan gratuito.

**Fijos, mientras la instancia está encendida**

| Recurso | Precio | Por día | Por mes (730 h) |
|---|---|--:|--:|
| Instancia `c7i-flex.large` | USD 0,08479 por hora | USD 2,03 | USD 61,90 |
| Disco gp3 de 30 GB | USD 0,08 por GB al mes | USD 0,08 | USD 2,40 |
| IP elástica (IPv4 pública) | USD 0,005 por hora | USD 0,12 | USD 3,65 |
| **Total** | | **USD 2,23** | **USD 67,95** |

Con la instancia detenida solo se cobran el disco y la IP elástica: unos USD 0,20
al día.

**Variables, según el uso**

| Recurso | Precio | Ejemplo |
|---|---|---|
| Nova Pro en Bedrock | USD 0,80 por millón de tokens de entrada y 3,20 de salida | ≈ USD 0,0025 por consulta |
| Ministral 3 8B en Bedrock | USD 0,15 por millón de tokens de entrada y de salida | ≈ USD 0,00045 por consulta |
| Almacenamiento del bucket (1,6 GB) | USD 0,023 por GB al mes | ≈ USD 0,04 al mes |
| Descargas públicas del bucket | Gratis los primeros 100 GB al mes de la cuenta; después, USD 0,09 por GB | `modelos.zip` (0,49 GB): ≈ USD 0,04 por descarga pasados los 100 GB |

Los modelos de lenguaje tienen un tope de USD 5 al día en total
(`SCIF_LLM_DAILY_BUDGET_USD`) y 5 consultas por minuto por IP: el peor caso es
USD 150 al mes. SciBERT y TF-IDF corren en la instancia y no suman costo por
consulta.

**Peor caso diario:** USD 2,23 de infraestructura más USD 5 de modelos de lenguaje,
unos USD 7,2 al día.

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
aws iam delete-role-policy --role-name scif-ec2-dvc-read --policy-name scif-bedrock-invoke
aws iam delete-role --role-name scif-ec2-dvc-read
```

El bucket de DVC no se borra con estos comandos: guarda el dataset y los modelos.
Tiene lectura pública en `dvcstore/` (los archivos de DVC) y en `descargas/` (los
zips del README), sin permiso para listar ni escribir. Para volverlo privado:

```bash
aws s3api delete-bucket-policy --bucket scif-dvc-e645d210
aws s3api put-public-access-block --bucket scif-dvc-e645d210 --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

## Historial

Los scripts de la entrega 2 (servidor de MLflow y entrenamiento en GPU en el
AWS Academy Learner Lab) están en [`historial/`](historial/README.md).

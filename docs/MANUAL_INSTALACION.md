# Manual de instalación del tablero

Prototipo de clasificación de la función retórica de citas científicas y recuperación
de pasajes del artículo citado. Consta de dos servicios:

| Servicio | Qué hace | Puerto |
|---|---|---|
| `api` | Inferencia: clasificación y recuperación de pasajes | 8000 (interno) |
| `dashboard` | Sirve el tablero y hace de proxy hacia la API | **8080** (público) |

Solo se publica el puerto del tablero. El navegador habla con un único origen, así que
no hay que configurar CORS ni abrir un segundo puerto.

---

## Opción A — Docker (recomendada)

### Requisitos

- Docker Engine 24+ con el plugin `docker compose`
- **2 GB de RAM libres** como mínimo (el servicio de inferencia usa ~950 MB)
- 3 GB de disco
- Salida a internet hacia `arxiv.org` (la recuperación descarga el PDF citado)

### Pasos

```bash
git clone https://github.com/MAIA-ProyectoGrado/pds-proyecto-tema1.git
cd pds-proyecto-tema1
```

Traer los pesos del modelo SciBERT, que se versionan con DVC y no con Git:

```bash
pip install 'dvc[s3]'
dvc pull models/scif-scibert.dvc
```

> Si no hay credenciales de S3, el sistema **igual funciona**: arranca con la línea base
> TF-IDF y el tablero indica que la recuperación de pasajes no está disponible. La línea
> base sí viaja en el repositorio.

Levantar:

```bash
docker compose up -d --build
```

La primera construcción descarga PyTorch y tarda entre 3 y 8 minutos. Al terminar, el
tablero queda en **http://localhost:8080**.

### Comprobar que quedó bien

```bash
docker compose ps                       # ambos servicios "healthy"
curl http://localhost:8080/api/models   # lista los modelos disponibles
```

La respuesta debe incluir `"retrieval_available": true` si los pesos de SciBERT están
presentes.

### Operación

```bash
docker compose logs -f api     # registros del servicio de inferencia
docker compose restart         # reiniciar
docker compose down            # detener (no borra imágenes ni pesos)
SCIF_PORT=9000 docker compose up -d    # publicar en otro puerto
```

---

## Opción B — Despliegue en AWS EC2

### Recursos necesarios

| Recurso | Mínimo | Recomendado |
|---|---|---|
| RAM | 2 GB | 4 GB |
| Disco | 10 GB | 16 GB |
| CPU | 2 vCPU | 2 vCPU |

El servicio de inferencia usa alrededor de **950 MB de memoria**, así que 1 GB no
alcanza: la instancia se queda sin memoria al cargar el modelo. En disco pesan la imagen
(~450 MB en x86_64) y los pesos (~420 MB), más el margen del sistema operativo.

AMI: Ubuntu Server 22.04 o 24.04.

**Grupo de seguridad:** abrir `8080/tcp` de entrada y permitir la salida a internet
(HTTPS) para que la API pueda descargar artículos de arXiv.

### Pasos

```bash
ssh -i <llave>.pem ubuntu@<IP-PUBLICA>
git clone https://github.com/MAIA-ProyectoGrado/pds-proyecto-tema1.git
cd pds-proyecto-tema1
bash infra/deploy_ec2.sh
```

El script se encarga de todo: instala Docker si falta, intenta `dvc pull` de los pesos,
construye las imágenes, levanta los servicios y espera a que el healthcheck pase. Al
terminar imprime la URL pública.

> La primera vez, tras instalar Docker el script pide cerrar la sesión SSH y volver a
> entrar (para que el usuario quede en el grupo `docker`), y ejecutarlo de nuevo.

**Las imágenes se construyen en la propia instancia a propósito.** Los equipos del
equipo son arm64 y la EC2 es x86_64; una imagen construida localmente no arrancaría allí.

### Verificar

```bash
curl http://<IP-PUBLICA>:8080/api/models
```

Y abrir `http://<IP-PUBLICA>:8080` en el navegador.

### Al terminar la evaluación

```bash
docker compose down     # detiene los contenedores
```

Detener la instancia desde la consola de AWS (**Stop**, no *Terminate*): la entrega pide
conservar las máquinas para poder relanzarlas.

---

## Opción C — Sin Docker (desarrollo)

Requiere **Python 3.12**: PyTorch no publica ruedas para 3.13 ni 3.14.

```bash
python3.12 -m venv .venv-api
source .venv-api/bin/activate
pip install -r src/api/requirements.txt
```

En dos terminales, desde la raíz del repositorio:

```bash
uvicorn src.api.main:app --port 8000     # terminal 1
python src/dashboard/serve.py            # terminal 2 → http://127.0.0.1:8080
```

En este modo el tablero llama a la API en `http://127.0.0.1:8000` (definido en
`src/dashboard/config.js`) y la API acepta ese origen por CORS.

---

## Variables de entorno

| Variable | Por defecto | Para qué |
|---|---|---|
| `SCIF_MODELS_DIR` | `models/` | Ruta de los pesos. Útil si están fuera del repositorio |
| `SCIF_ALLOWED_ORIGINS` | `http://localhost:8080,http://127.0.0.1:8080` | Orígenes que acepta CORS. Irrelevante con Docker (mismo origen) |
| `SCIF_PORT` | `8080` | Puerto público del tablero |

---

## Problemas frecuentes

**El tablero dice «API no disponible».**
La API no responde. Revisar `docker compose ps` y `docker compose logs api`. Si el
servicio aparece como `unhealthy`, lo más probable es falta de memoria: comprobar con
`docker stats` y usar una instancia con más memoria.

**El panel de pasajes muestra «pesos no encontrados» o error 503.**
Faltan los pesos de SciBERT en `models/scif-scibert/`. Ejecutar
`dvc pull models/scif-scibert.dvc`, o copiar la carpeta con `scp`, y reiniciar.
La clasificación con la línea base sigue funcionando sin ellos.

**Error 502 al recuperar pasajes.**
La instancia no alcanza `arxiv.org`. Revisar las reglas de salida del grupo de
seguridad. Alternativa: usar la opción «Pegar el texto en su lugar» del tablero.

**La primera consulta tarda 10–15 segundos.**
Es la carga inicial del encoder. El servicio la hace en segundo plano al arrancar; si se
consulta de inmediato, hay que esperar. Las siguientes responden en milisegundos.

**`docker compose build` falla por falta de espacio.**
`docker builder prune -af` libera la caché de construcciones anteriores.

---

## Pruebas automatizadas

```bash
pip install -r requirements-dev.txt
pytest -q
```

38 pruebas. Las que necesitan SciBERT se omiten automáticamente si faltan `torch` o los
pesos, de modo que la integración continua no falla por ello.

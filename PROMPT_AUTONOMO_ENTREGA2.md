# Runbook — Entrega 2 (SCIF): partes 2 y 3

> **Cómo usar este archivo.** Es la guía de ejecución para la máquina con GPU
> (git, credenciales AWS, dvc y python ya configurados). Los pasos son secuenciales
> y solo requieren intervención manual en las 3 pausas marcadas `PANTALLAZO`.
> El trabajo de la parte 1 y todos los scripts ya están en la rama
> `feature/entrega2-eda-modelos` del repo.

---

## 0. Contexto (no re-derivar)

- **Proyecto:** SCIF — clasificación de la función retórica de una cita (9 clases)
  y (a futuro) recomendación local de fragmentos del artículo citado.
- **Repo:** `https://github.com/alvarorf/pds-proyecto-tema1`
  → rama de trabajo: **`feature/entrega2-eda-modelos`** (ya creada, con datos y scripts).
- **Bucket DVC (S3):** `s3://citation-dvcstore-tema1` (`arn:aws:s3:::citation-dvcstore-tema1`), región **us-east-1**.
- **Presupuesto AWS:** quedan **~USD 20**. La instancia de entrenamiento puede
  usarse **máx. 15 h**. Al terminar: **detener (stop), NO terminar (terminate)** la EC2.
- **Fecha de entrega:** domingo 6 de septiembre (semana 5).
- **Qué ya está hecho (parte 1 y empaquetado):**
  - `dataset/` → 4 CSV enriquecidos (train 12 600 / val 2 700 / test 2 700 / master 18k).
  - `scripts/prepare_dataset.py` → estandariza y verifica no-leakage → `data/raw/*.csv|jsonl` + `data/processed/dataset_manifest.json`.
  - `scripts/eda_entrega2.py` → EDA orientado al feedback → `docs/eda_entrega2/`.
  - `scripts/train.py` → v1 (TF-IDF+LogReg) y v2 (SciBERT fine-tune) con MLflow.
  - `scripts/analyze_metrics.py` → tabla comparativa + sobreajuste → `docs/MODELOS_ENTREGA2.md`.
  - `infra/` → `launch_ec2.sh`, `mlflow_user_data.sh`, `remote_setup.sh`, `stop_ec2.sh`.
  - `docs/EDA_ENTREGA2.md` (redactado con números reales) y `MEJORAS_ENTREGA2.md`.
  - La API (`api/`) ya empaqueta el mejor modelo — **no tocar**.

### Feedback de la Entrega 1 que hay que cerrar (14+15+8+20+18+10 = 85/100)

| Ítem | Nota | Qué pidieron |
|---|---|---|
| Problema y contexto | 14/15 | «aterrizar la utilidad … definir el alcance» |
| Datos | 8/10 | «el archivo no contiene contextos de cita reales» |
| Exploración | **20/30** | «no se explora la variable objetivo (`base_label`) ni su balance entre las 9 categorías» |
| Maqueta | 18/20 | «la maqueta asume datos que no se mostraron consistentes» |

El EDA y `MEJORAS_ENTREGA2.md` ya responden a esto; tu trabajo es **generar la
evidencia** (MLflow, métricas, figuras) y **dejar los .md listos para pegar**.

---

## 1. Preparación local (≈ 10–15 min)

```bash
git clone https://github.com/alvarorf/pds-proyecto-tema1
cd pds-proyecto-tema1
git checkout feature/entrega2-eda-modelos

python -m venv .venv-train
source .venv-train/bin/activate           # Windows: .venv-train\Scripts\activate
pip install -r requirements-train.txt

aws sts get-caller-identity                # confirma credenciales + anota el "Arn"/usuario
aws configure get region                   # debe ser us-east-1 (o exporta AWS_DEFAULT_REGION=us-east-1)
```

Verifica que puedes escribir en el bucket:

```bash
aws s3 ls s3://citation-dvcstore-tema1/
```

---

## 2. Servidor de experimentos (EC2 + MLflow) (≈ 15–25 min)

### 2.1 Lanzar la instancia

> **Restricción confirmada del entorno (AWS Academy Learner Lab, política
> `Pvoclabs2`):** `ec2:RunInstances` **solo permite tamaños `≤ *.large` (2 vCPU)**;
> **todas las instancias GPU están denegadas**. No hay fine-tuning en GPU en esta
> cuenta. `ssm:GetParameters` de alias de AMI también está bloqueado (el script
> resuelve la AMI con `describe-images`). Se usa el key pair preexistente
> **`vockey`** (archivo local `llave.pem`) y el perfil **`LabInstanceProfile`**
> (le da a la EC2 credenciales de S3 propias, independientes de la sesión del lab).

| Tipo | vCPU / RAM | USD/h | Nota |
|---|---|--:|---|
| **`t3.large`** (por defecto) | 2 / 8 GB | ~0.083 | recomendado: la RAM importa para el fine-tuning CPU |
| `c5.large` | 2 / 4 GB | ~0.085 | alternativa; riesgo de OOM con SciBERT |

```bash
./infra/launch_ec2.sh                       # t3.large + vockey + LabInstanceProfile
```

El script:
- usa el key pair **`vockey`** y el perfil **`LabInstanceProfile`**;
- crea el Security Group `scif-mlflow-sg` **abriendo 22 y 5000 solo a tu IP**
  (esto ES "habilitar el puerto en las reglas de entrada del Security Group");
- lanza 1 instancia con `shutdown-behavior=stop` y un cron de **auto-stop a las 14 h**;
- instala MLflow 3.15.2 como servicio systemd (`sqlite` + artefactos locales) en `:5000`;
- escribe `infra/ec2_instance.env` con `PUBLIC_IP`, `INSTANCE_ID`, `MLFLOW_TRACKING_URI`.

> A ~USD 0.083/h, el servidor puede quedar encendido las ~15 h por < USD 1.3. El
> costo real de la Entrega 2 lo domina el tiempo de entrenamiento CPU (ver §4.2).

### 2.2 Esperar a MLflow y verificar

```bash
source infra/ec2_instance.env
curl -sf "$MLFLOW_TRACKING_URI/health" && echo "MLflow OK"   # reintenta 3-5 min
```

### 2.3 PANTALLAZO (bloque de capturas exigido por el enunciado)

Toma **3 capturas** y guárdalas en `docs/soportes/`:

1. **`docs/soportes/01_ec2_consola.png`** — consola de AWS → EC2 → Instances, con la
   fila de la instancia seleccionada mostrando **Instance ID**, **Public IPv4**,
   **Instance type** y (barra superior derecha) el **usuario/rol IAM** conectado.
2. **`docs/soportes/02_security_group.png`** — la pestaña *Inbound rules* del SG
   `scif-mlflow-sg` con el puerto **5000** y **22** visibles.
3. **`docs/soportes/03_mlflow_ui.png`** — navegador en `http://<PUBLIC_IP>:5000`
   con **la IP pública visible en la barra de direcciones** y el experimento
   `scif-citation-intent` (después de entrenar, que se vean los runs).

> Toma la #3 otra vez al final, cuando ya haya runs v1 y v2 con métricas.

---

## 3. Datos: subir la nueva versión a S3 y versionar (≈ 10–20 min)

Desde tu **máquina local** (tiene los CSV en `dataset/`):

```bash
# 1. estandariza y valida (genera data/raw/*.csv|jsonl y el manifest)
python scripts/prepare_dataset.py --src dataset --out_raw data/raw --out_proc data/processed

# 2. versiona con DVC y sube al bucket
dvc add data/raw
git add data/raw.dvc data/.gitignore data/processed/dataset_manifest.json
git commit -m "data(entrega2): dataset enriquecido de intencion de cita (9 clases reales) versionado con DVC"
dvc push                      # sube ~16 MB a s3://citation-dvcstore-tema1
git push origin feature/entrega2-eda-modelos
```

> `data/raw.dvc` pasa de apuntar al corpus viejo (2 archivos de abstracts) al
> nuevo (`master`, `train/val/test` en csv+jsonl). Ese cambio de hash **es** la
> "nueva versión de los datos".

---

## 4. Entrenamiento y registro (parte 3) (ver §7 para tiempos)

### 4.1 En la EC2: traer repo + datos

```bash
ssh -i infra/scif-entrega2.pem ubuntu@$PUBLIC_IP
# dentro de la EC2:
export REPO_URL=https://github.com/alvarorf/pds-proyecto-tema1
export BRANCH=feature/entrega2-eda-modelos
curl -sL "https://raw.githubusercontent.com/alvarorf/pds-proyecto-tema1/$BRANCH/infra/remote_setup.sh" -o remote_setup.sh
bash remote_setup.sh
```

`remote_setup.sh` hace: `git checkout` de la rama, venv + `requirements-train.txt`,
**`dvc pull`** (baja los datos del bucket), corre el **EDA**, y lanza **v1** y **v2**
apuntando a `MLFLOW_TRACKING_URI=http://localhost:5000`.

> Las credenciales AWS en la EC2: usa un **IAM role** en la instancia o copia
> `~/.aws/credentials` por `scp`. `dvc pull` necesita permiso de lectura sobre el bucket.

### 4.2 Las dos iteraciones (ya configuradas en `scripts/train.py`)

| Versión | Modelo | Config (CPU 2 vCPU) | Rol |
|---|---|---|---|
| **v1 baseline** | TF-IDF (1–2 gramas) + Regresión Logística `class_weight=balanced` | `C=1.0`, `max_features=50k` | referencia; **sobreajusta severo** (F1-macro val **0.496**, brecha train→val **0.30**, medido) |
| **v2 (iteración intermedia)** | fine-tuning `distilbert-base-uncased` (SciBERT si la RAM aguanta) | `--max_train 5000 --epochs 1 --max_len 128 --batch 16` | mejora esperada de F1-macro val; **no optimizar más** — deja hueco para v3 |

> **Por qué DistilBERT y submuestreo:** el Learner Lab solo da 2 vCPU sin GPU.
> SciBERT completo (12 600 filas × 3 épocas) tomaría 10–20 h. `distilbert` con
> `--max_train 5000 --epochs 1 --max_len 128` entrena en **~1.5–3 h** y sigue
> siendo un v2 legítimo (encoder contextual > TF-IDF). Si sobra tiempo/RAM, subir a
> `--max_train 9000 --epochs 2` o cambiar a SciBERT.
>
> **v2 NO debe ser perfecto.** Nada de búsqueda de hiperparámetros ni ensembles.
> Si v2 val F1-macro cae en ~0.55–0.68, es suficiente; el resto se documenta como v3.

Comandos (si no usas `remote_setup.sh`):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export SCIF_SKIP_MODEL_LOGGING=0        # 1 solo para pruebas rapidas sin registrar el artefacto
python scripts/train.py --stage v1 --data_dir data/raw
python scripts/train.py --stage v2 --data_dir data/raw \
    --model distilbert-base-uncased --max_train 5000 --epochs 1 --max_len 128 --batch 16 --lr 3e-5
```

Alternativa **sin fine-tuning** (si el CPU no alcanza): embeddings de
`sentence-transformers/all-MiniLM-L6-v2` (calcular una vez, ~20–40 min) + Regresión
Logística. Es un v2 defendible y más rápido; queda como plan B documentado.

Cada run registra en MLflow: params, `train/val/test` de accuracy + F1 (macro/micro),
F1 por clase, matrices de confusión normalizadas, `fit_seconds`, brechas
`overfit_gap_*` y un tag `overfitting ∈ {bajo, moderado, severo}`. Ambos modelos
quedan en el Model Registry como `scif-citation-intent` (v1 = versión 1, v2 = versión 2).

### 4.3 Análisis de métricas, error y sobreajuste

```bash
python scripts/analyze_metrics.py --results docs/model_results.json --out docs/MODELOS_ENTREGA2.md
```

Luego **edita `docs/MODELOS_ENTREGA2.md`** y completa la sección "Lectura de
resultados" con lo que muestren los runs:
- confirma/ajusta la brecha train→val de cada modelo y su veredicto;
- nombra las 2–3 parejas de clases más confundidas (matriz de confusión de v2);
- comprueba que la brecha val→test es pequeña (señal de que el split es sano);
- concluye explícitamente: **v1 sobreajusta**, **v2 generaliza mejor pero deja
  margen para v3**, y **el balanceo del dataset es artificial** (validez externa limitada).

### 4.4 Cierre AWS

```bash
# baja artefactos de MLflow que quieras conservar (opcional):
#   scp -i infra/scif-entrega2.pem -r ubuntu@$PUBLIC_IP:/opt/mlflow/mlflow.db docs/soportes/
./infra/stop_ec2.sh          # DETIENE (no termina) la instancia
```

Vuelve a tomar `docs/soportes/03_mlflow_ui.png` con los runs v1 y v2 visibles.
Verifica en la consola de Billing que el gasto incremental sea el esperado (< USD 16).

---

## 5. Documento para la entrega — QUÉ generar y DÓNDE va cada cambio

Al terminar debe quedar **`MEJORAS_ENTREGA2.md`** (ya existe un borrador en el repo)
**actualizado y listo para pegar** en el reporte de la Entrega 2. Ese archivo ya
contiene el mapa exacto *feedback → sección/página/párrafo del PDF de la Entrega 1
→ texto nuevo*. Tu tarea es:

1. Rellenar los `⟨...⟩` de `MEJORAS_ENTREGA2.md` con los **números reales** de los
   runs (F1-macro de v1 y v2, brechas de sobreajuste, clases confundidas).
2. Pegar 2–3 figuras de `docs/eda_entrega2/` y las matrices de confusión de MLflow
   en las secciones que las piden.
3. No superar **10 páginas** en el reporte final (regla del enunciado): el reporte
   de la Entrega 2 es nuevo y corto; `MEJORAS_ENTREGA2.md` indica qué párrafos de
   la Entrega 1 se reemplazan y cuáles se resumen.

**Resumen del mapa (el detalle con texto propuesto está en `MEJORAS_ENTREGA2.md`):**

| # | Feedback | Ubicación en "Entrega 1 Proyectos.pdf" | Acción |
|---|---|---|---|
| A | "aterrizar la utilidad / alcance" | **pág. 2**, "PROBLEMA A ABORDAR Y CONTEXTO", último párrafo; **pág. 3**, "ALCANCE DEL PROYECTO", viñeta *Datos* y viñeta *Limitaciones* | Reemplazar el cierre genérico por 1 párrafo de **alcance concreto** (usuario, entrada/salida, qué NO hace) |
| B | "el archivo no contiene contextos de cita reales" | **pág. 4–5**, "DESCRIPCIÓN DEL CONJUNTO DE DATOS" y "ESTRUCTURACIÓN DEL DATASET" (lista de variables) | Sustituir por la descripción del **dataset enriquecido** (12 600/2 700/2 700, 9 clases, `citation_context` real con mediana 26 palabras, tabla de variables nueva) |
| C | "no se explora la variable objetivo ni su balance entre las 9 categorías" | **pág. 5–8**, "EXPLORACIÓN DE LOS DATOS" y "Conclusiones" (reemplaza el análisis de los abstracts) | Insertar **§2 del EDA** (distribución de `label`, ratio 1.04, entropía 1.00, figura `01_variable_objetivo.png`) como **primer** bloque de la exploración |
| D | "la maqueta asume datos que no se mostraron consistentes" | **pág. 8–10**, "MAQUETA DEL PROTOTIPO" (bloque *Resultados* y *Top-3 de pasajes recuperados*) | Añadir nota: el panel Top-3 queda **marcado como ilustrativo/pendiente de datos**; la maqueta de la Entrega 2 se centra en el clasificador de 9 clases (evidencia §4 del EDA) |
| E | soporte MLflow/EC2 (requisito nuevo) | Reporte Entrega 2, sección "Experimentos" | Insertar `docs/soportes/01..03` + tabla de `docs/MODELOS_ENTREGA2.md` |
| F | trabajo en equipo | Reporte Entrega 2, última página | Actualizar tabla de responsables incluyendo EDA v2 / EC2 / modelos |

---

## 6. Checklist final (Definition of Done)

- [ ] `data/raw.dvc` actualizado y `dvc push` hecho (bucket tiene la versión nueva).
- [ ] Rama `feature/entrega2-eda-modelos` pusheada con: datos versionados, `docs/EDA_ENTREGA2.md`, `docs/MODELOS_ENTREGA2.md`, `docs/eda_entrega2/*`, `docs/soportes/*`, `MEJORAS_ENTREGA2.md` actualizado.
- [ ] MLflow con **≥ 2 runs** (v1, v2), métricas train/val/test, matrices de confusión y tags de overfitting.
- [ ] 3 pantallazos en `docs/soportes/` (EC2 con IP+usuario, SG con puerto 5000, MLflow UI con IP).
- [ ] EC2 en estado **stopped** (no terminated). `aws ec2 describe-instances ... State.Name == stopped`.
- [ ] Gasto AWS incremental verificado (< USD 16; idealmente < USD 6).
- [ ] `MEJORAS_ENTREGA2.md` sin `⟨placeholders⟩`, listo para pegar en el reporte.
- [ ] PR de la rama a `main` abierto (no *merge* hasta revisión del equipo).

---

## 7. Estimación de tiempos

### a) Ejecutar este prompt de principio a fin

| Fase | GPU (g4dn/g5) | CPU (c7i) |
|---|--:|--:|
| Prep local + venv | 10–15 min | 10–15 min |
| Lanzar EC2 + MLflow arriba + verificar | 15–25 min | 15–25 min |
| 3 pantallazos (manual) | ~5 min | ~5 min |
| `prepare_dataset` + `dvc add/push` + `git push` | 10–20 min | 10–20 min |
| `dvc pull` en EC2 + instalar deps | 8–15 min | 8–15 min |
| EDA | 3–5 min | 3–5 min |
| Entrenar v1 | 1–2 min | 1–2 min |
| Entrenar v2 | 20–45 min | 1.5–3 h |
| `analyze_metrics` + redacción de conclusiones | 15–25 min | 15–25 min |
| Actualizar `MEJORAS_ENTREGA2.md` + figuras + pantallazo final | 20–30 min | 20–30 min |
| Stop EC2 + verificación de costos | ~5 min | ~5 min |
| **Total (wall-clock)** | **≈ 2 – 3 h** | **≈ 3.5 – 5.5 h** |
| *de eso, atención humana activa* | ~45–60 min | ~45–60 min |

### b) Solo el entrenamiento (v1 + v2)

| | v1 | v2 | Total |
|---|--:|--:|--:|
| GPU T4 (g4dn.xlarge) | ~1 min | 25–45 min | **~30–50 min** |
| GPU A10G (g5.xlarge) | ~1 min | 12–25 min | **~15–30 min** |
| CPU (c7i.2xlarge, 8 vCPU) | ~1 min | 1.5–3 h | **~1.5–3 h** |

> Datos de referencia: train 12 600 filas, contexto mediana 26 palabras
> (`max_len=256` sobra), 9 clases balanceadas, 3 épocas ≈ 2 364 pasos con batch 16.

---

## 8. Notas de seguridad / gotchas

- **Rotar credenciales:** el archivo local `AWS CLI.txt` de la carpeta padre tiene
  llaves y una private key en claro. No las comitas y pide rotarlas al terminar.
- El `.pem` (`llave.pem`, `infra/*.pem`) está en `.gitignore` (`*.pem`) — mantenlo así.
- Si `mlflow` lanza *"filesystem tracking backend in maintenance mode"* al correr
  local: es esperado; en la EC2 se usa backend `sqlite`, no aplica.
- Si no hay cuota de GPU y tampoco quieres CPU lento: `--model distilbert-base-uncased
  --epochs 2 --max_len 192` entrena en ~20–30 min en 8 vCPU y sigue siendo un v2 válido.
- `transformers` puede pedir `accelerate`; ya está en `requirements-train.txt`.
- El auto-stop del `user_data` apaga la instancia a las 14 h de uptime — si necesitas
  más, borra `/etc/cron.d/auto-stop` (pero recuerda el techo de 15 h del presupuesto).

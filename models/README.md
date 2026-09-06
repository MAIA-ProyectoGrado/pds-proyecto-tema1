# Modelos empaquetados (listos para servir)

Cada carpeta es un modelo autocontenido para la clasificación de la **función de
cita** (9 clases). La API de inferencia (`api/`) solo tiene que apuntar a una de
ellas — no requiere cambios de código, basta cargar el artefacto.

**Entrada esperada por todos los modelos:**
`citation_context + " [SEC] " + <sección retórica canónica>`
(la normalización de la sección está en `models/scif_predict.py::canon_section`).

**Orden de las 9 clases** (usado en entrenamiento, alfabético):
`Application, Background, Basis, Comparison, Evidence, Further_Reading, Gap,
Identification_of_the_Originator, Modification_Improvement`.

---

## `scif-v1-tfidf-logreg/`  — línea base (disponible)

Pipeline **TF-IDF (1–2 gramas) + Regresión Logística** entrenado sobre el dataset
v3 completo (14.461 ejemplos). Formato MLflow *sklearn* (`model.pkl`, cloudpickle;
`scikit-learn==1.7.2`). ~5 MB, sin GPU.

- F1-macro: validación **0.516**, prueba 0.518.

## `scif-scibert/`  — encoder de dominio científico (mejor modelo, pendiente de empaquetar)

`v3_scibert_finetune` alcanzó **F1-macro 0.640 / 0.634 (val / test)** y es el mejor
modelo del proyecto (métricas, curvas y matrices en MLflow y en
`docs/MODELOS_ENTREGA2.md`). Su **entrenamiento quedó registrado completo** pero los
pesos no se serializaron en esa corrida (fallo al guardar el checkpoint; corregido
en `scripts/train.py`, que ahora escribe la carpeta con `--model_out`).

Para regenerarlo (deja la carpeta lista en `models/scif-scibert/`):

```bash
python scripts/train.py --stage v2 --data_dir data/raw --v2_method finetune \
  --model allenai/scibert_scivocab_uncased \
  --epochs 4 --batch 32 --max_len 256 --eval_steps 150 \
  --model_out models/scif-scibert
```

~20–35 min en una GPU `g5.xlarge`; ~1.5–3 h en CPU de 8+ vCPU. También se puede
recuperar el checkpoint del *snapshot* EBS `snap-0e39c9347de9a4423`.

---

## Cómo lo consume la API

`models/scif_predict.py` es una implementación de referencia del cargador
(autodetecta *sklearn* vs. *transformer*). La API puede importarla o replicar su
lógica:

```python
from models.scif_predict import Predictor
p = Predictor("models/scif-v1-tfidf-logreg")     # o "models/scif-scibert"
p.predict("... citation context ...", rhetorical_section="Introduction")
# -> {predicted_label, confidence, probabilities, ambiguous}
```

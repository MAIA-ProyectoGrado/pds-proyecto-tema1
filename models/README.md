# Modelos

Artefactos que sirve la API. Cada carpeta es autocontenida: la API solo necesita
la ruta (`SCIF_MODELS_DIR`) y no hay que cambiar código para cargarlos.

**Orden de las 9 clases** (alfabético, el mismo del entrenamiento):
`Application, Background, Basis, Comparison, Evidence, Further_Reading, Gap,
Identification_of_the_Originator, Modification_Improvement`.

| Carpeta | Qué es | Tamaño | Versionado |
|---|---|--:|---|
| `scif-scibert-chunks/` | SciBERT configuración B (v4), el modelo recomendado | 440 MB | DVC |
| `minilm-l6-v2/` | Codificador que elige las 3 oraciones clave del artículo citado | 87 MB | DVC |
| `scif-v4-tfidf-logreg/` | TF-IDF + regresión logística (v4), la línea base liviana | 4,8 MB | git |
| `archivo/` | Modelos de la entrega 2, que la API no sirve | | git y DVC |

La evolución completa y las métricas están en `docs/HISTORIAL_MODELOS.md`.

## `scif-scibert-chunks/`: SciBERT configuración B

`AutoModelForSequenceClassification` sobre `allenai/scibert_scivocab_uncased`, en
formato estándar de Hugging Face (`model.safetensors`, tokenizer, `label_order.json`).
Se obtuvo del checkpoint del entrenamiento con `scripts/convert_checkpoint.py`, que
carga solo los pesos.

Lee un **par de textos**, con un máximo de 512 tokens (`input_format.json`):

```text
texto_a = <contexto de cita>
texto_b = Section: <sección> | Cited passages: <oración 1> <oración 2> <oración 3>
```

- La sección va tal cual; si no se conoce, se omite esa parte.
- Las oraciones clave son las 3 del abstract del artículo citado más parecidas al
  contexto según `minilm-l6-v2` (`models/passages.py`).
- Sin artículo citado, `texto_b` termina en `Passages unavailable`.

La construcción del par está en `models/scif_predict.py::build_pair_config_b`, la
misma función que usa la evaluación (`scripts/evaluate_v4.py`).

## `minilm-l6-v2/`: codificador de oraciones

`sentence-transformers/all-MiniLM-L6-v2` guardado para uso sin conexión con
`scripts/download_encoder.py`. `models/passages.py` reproduce su cálculo de vectores
(promedio de tokens y normalización L2, 256 tokens) sin depender de
`sentence-transformers`. Sobre 300 filas de validación elige exactamente las mismas
oraciones que el dataset de entrenamiento.

## `scif-v4-tfidf-logreg/`: línea base

Pipeline de scikit-learn (`model.pkl`, cloudpickle, `scikit-learn==1.7.2`) entrenado
con `scripts/train_tfidf_v4.py`. Lee `contexto [SEC] sección normalizada`
(`models/scif_predict.py::build_input`) y no usa el artículo citado. Responde en
milisegundos y no necesita torch.

## Uso fuera de la API

```python
from models.passages import SentenceEncoder, select_passages
from models.scif_predict import Predictor

clf = Predictor("models/scif-scibert-chunks")
enc = SentenceEncoder("models/minilm-l6-v2")

contexto = "We fine-tune SciBERT [4] on our citation corpus."
abstract = "..."  # abstract del artículo citado
oraciones = [p.text for p in select_passages(contexto, abstract, enc, k=3)]
clf.predict(contexto, "Methods", passages=oraciones)
# -> {predicted_label, confidence, probabilities, ambiguous, model_input}
```

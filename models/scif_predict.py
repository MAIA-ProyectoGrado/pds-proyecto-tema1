"""
Cargador de referencia para los modelos empaquetados de SCIF.

Autodetecta el formato de la carpeta del modelo:
  * `model.pkl`      -> pipeline sklearn (TF-IDF v4)
  * `config.json`    -> encoder Hugging Face ajustado. Si `input_format.json` dice
                        `"input": "pair"` (SciBERT configuración B, v4), lee el par
                        contexto / sección + oraciones clave; si no, un solo texto.

Uso:
    from models.scif_predict import Predictor
    p = Predictor("models/scif-scibert-chunks")
    p.predict("We fine-tune SciBERT [4].", "Methods", passages=["...", "...", "..."])

No depende de la API; la API lo importa.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

LABELS = [
    "Application", "Background", "Basis", "Comparison", "Evidence",
    "Further_Reading", "Gap", "Identification_of_the_Originator",
    "Modification_Improvement",
]


def canon_section(s: str | None) -> str:
    """Normaliza la sección retórica a una categoría canónica (entrada del TF-IDF v4)."""
    if not isinstance(s, str) or not s.strip():
        return "Desconocida"
    s2 = s.strip().lower()
    for key, lab in [
        ("abstract", "Abstract"), ("introduction", "Introduccion"),
        ("related work", "Trabajo_relacionado"), ("background", "Trabajo_relacionado"),
        ("method", "Metodos"), ("approach", "Metodos"),
        ("experiment", "Experimentos"), ("result", "Resultados"),
        ("discussion", "Discusion"), ("conclusion", "Conclusion"),
        ("scientific body", "Cuerpo_generico"),
    ]:
        if key in s2:
            return lab
    return "Otra"


def build_input(citation_context: str, rhetorical_section: str | None) -> str:
    text = re.sub(r"\s+", " ", str(citation_context)).strip()
    return f"{text} [SEC] {canon_section(rhetorical_section)}"


def _valid(value) -> bool:
    """Criterio del notebook de entrenamiento: texto no vacío y distinto de 'nan'."""
    s = str(value).strip() if value is not None else ""
    return bool(s) and s.lower() != "nan"


def build_pair_config_b(citation_context: str, rhetorical_section: str | None,
                        passages: list[str] | None = None,
                        cited_title: str | None = None) -> tuple[str, str]:
    """Entrada de la configuración B (modelo v4), idéntica a `construir_entrada_config_b`
    del notebook de entrenamiento.

    Devuelve el par (texto_a, texto_b) que se tokeniza como `tokenizer(a, b)`:
      texto_a = contexto de cita
      texto_b = "Section: <sección> | Cited passages: <oración1> <oración2> <oración3>"
    Sin pasajes, texto_b usa "Cited work: <título>" o "Passages unavailable".
    La sección se pasa tal cual, sin normalizar, como en el entrenamiento.
    """
    ctx = str(citation_context).strip() if citation_context is not None else ""
    parts = []
    if _valid(rhetorical_section):
        parts.append(f"Section: {str(rhetorical_section).strip()}")
    valid = [str(p).strip() for p in (passages or []) if _valid(p)]
    if valid:
        parts.append("Cited passages: " + " ".join(valid))
    elif _valid(cited_title):
        parts.append(f"Cited work: {str(cited_title).strip()}")
    else:
        parts.append("Passages unavailable")
    return ctx, " | ".join(parts)


def model_kwargs(encoding, input_format: dict) -> dict:
    """Entradas del tokenizer que recibe el modelo.

    `input_format["model_inputs"]` lista las que usó el entrenamiento. En la
    configuración B son `input_ids` y `attention_mask`: sin `token_type_ids`,
    BERT trata el par completo como un solo segmento. Si la lista no está, el
    modelo recibe todo lo que produce el tokenizer.
    """
    keys = input_format.get("model_inputs")
    return {k: encoding[k] for k in keys} if keys else dict(encoding)


class Predictor:
    def __init__(self, model_dir: str | os.PathLike, ambiguity_threshold: float = 0.10):
        self.model_dir = Path(model_dir)
        self.classes = list(LABELS)
        self.ambiguity_threshold = ambiguity_threshold
        fmt_path = self.model_dir / "input_format.json"
        self.input_format = json.loads(fmt_path.read_text()) if fmt_path.exists() else {}
        if (self.model_dir / "model.pkl").exists():
            import cloudpickle
            with open(self.model_dir / "model.pkl", "rb") as f:
                self._pipe = cloudpickle.load(f)
            self.classes = list(getattr(self._pipe, "classes_", LABELS))
            self.kind = "sklearn"
        elif (self.model_dir / "config.json").exists():
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            self._model.eval()
            lo = self.model_dir / "label_order.json"
            self.classes = json.loads(lo.read_text()) if lo.exists() else [
                self._model.config.id2label[i] for i in range(self._model.config.num_labels)]
            self._max_len = self.input_format.get("max_len", 256)
            self.kind = "transformer"
        else:
            raise FileNotFoundError(f"{self.model_dir}: falta model.pkl o config.json")

    @property
    def reads_passages(self) -> bool:
        """El modelo lee las oraciones clave del artículo citado (configuración B)."""
        return self.kind == "transformer" and self.input_format.get("input") == "pair"

    def model_input(self, citation_context: str, rhetorical_section: str | None = None,
                    passages: list[str] | None = None,
                    cited_title: str | None = None) -> str | tuple[str, str]:
        """Entrada exacta que recibe el modelo: un texto, o el par (texto_a, texto_b)."""
        if self.reads_passages:
            return build_pair_config_b(citation_context, rhetorical_section, passages, cited_title)
        return build_input(citation_context, rhetorical_section)

    def predict(self, citation_context: str, rhetorical_section: str | None = None,
                passages: list[str] | None = None, cited_title: str | None = None) -> dict:
        x = self.model_input(citation_context, rhetorical_section, passages, cited_title)
        if self.kind == "sklearn":
            proba = self._pipe.predict_proba([x])[0]
        else:
            torch = self._torch
            args = x if isinstance(x, tuple) else (x,)
            enc = self._tok(*args, truncation=True, max_length=self._max_len, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**model_kwargs(enc, self.input_format)).logits
                proba = torch.softmax(logits[0], dim=-1).tolist()
        probs = {c: round(float(p), 4) for c, p in zip(self.classes, proba)}
        probs = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
        vals = list(probs.values())
        best = next(iter(probs))
        return {
            "predicted_label": best,
            "confidence": probs[best],
            "probabilities": probs,
            "ambiguous": len(vals) >= 2 and (vals[0] - vals[1]) < self.ambiguity_threshold,
            "model_input": x,
        }


if __name__ == "__main__":  # prueba rápida
    import sys
    p = Predictor(sys.argv[1] if len(sys.argv) > 1 else "models/scif-v4-tfidf-logreg")
    for ctx, sec in [
        ("Although several approaches exist, it remains an open problem [3].", "Introduction"),
        ("Our results are in strong agreement with Smith et al. (2020).", "Discussion"),
        ("We use the ADAM optimizer (Kingma & Ba, 2015).", "Methods"),
        ("For a comprehensive review, see Jones (2019).", None),
    ]:
        r = p.predict(ctx, sec)
        print(f"{r['predicted_label']:<32} conf={r['confidence']:.2f} amb={r['ambiguous']}  | {ctx[:55]}")

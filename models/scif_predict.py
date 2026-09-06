"""
Cargador de referencia para los modelos empaquetados de SCIF.

Autodetecta el formato de la carpeta del modelo:
  * `model.pkl`      -> pipeline sklearn (línea base v1)
  * `config.json`    -> encoder HuggingFace ajustado (v3: SciBERT / SPECTER2)

Uso:
    from models.scif_predict import Predictor
    p = Predictor("models/scif-v1-tfidf-logreg")
    p.predict("Our results agree with Smith et al. (2020).", rhetorical_section="Discussion")

No depende de la API; la API puede importar esto o replicar la lógica.
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
    """Normaliza la sección retórica a una de 10 categorías canónicas."""
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


class Predictor:
    def __init__(self, model_dir: str | os.PathLike):
        self.model_dir = Path(model_dir)
        self.classes = list(LABELS)
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
            fmt = self.model_dir / "input_format.json"
            self._max_len = json.loads(fmt.read_text()).get("max_len", 256) if fmt.exists() else 256
            self.kind = "transformer"
        else:
            raise FileNotFoundError(f"{self.model_dir}: falta model.pkl o config.json")

    def predict(self, citation_context: str, rhetorical_section: str | None = None) -> dict:
        text = build_input(citation_context, rhetorical_section)
        if self.kind == "sklearn":
            proba = self._pipe.predict_proba([text])[0]
        else:
            torch = self._torch
            enc = self._tok(text, truncation=True, max_length=self._max_len, return_tensors="pt")
            with torch.no_grad():
                proba = torch.softmax(self._model(**enc).logits[0], dim=-1).tolist()
        probs = {c: round(float(p), 4) for c, p in zip(self.classes, proba)}
        probs = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
        vals = list(probs.values())
        best = next(iter(probs))
        return {
            "predicted_label": best,
            "confidence": probs[best],
            "probabilities": probs,
            "ambiguous": len(vals) >= 2 and (vals[0] - vals[1]) < 0.10,
        }


if __name__ == "__main__":  # prueba rápida
    import sys
    p = Predictor(sys.argv[1] if len(sys.argv) > 1 else "models/scif-v1-tfidf-logreg")
    for ctx, sec in [
        ("Although several approaches exist, it remains an open problem [3].", "Introduction"),
        ("Our results are in strong agreement with Smith et al. (2020).", "Discussion"),
        ("We use the ADAM optimizer (Kingma & Ba, 2015).", "Methods"),
        ("For a comprehensive review, see Jones (2019).", None),
    ]:
        r = p.predict(ctx, sec)
        print(f"{r['predicted_label']:<32} conf={r['confidence']:.2f} amb={r['ambiguous']}  | {ctx[:55]}")

import random

CLASSES = [
    "Background", 
    "Gap", 
    "Basis", 
    "Comparison", 
    "Application", 
    "Modification", 
    "Evidence", 
    "Identification of the Originator", 
    "Further Reading"
]

class ModelPredictor:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.is_ready = True

    def predict(self, text: str):
        probs = {label: round(random.uniform(0.01, 0.15), 4) for label in CLASSES}
        best_label = "Gap"
        probs[best_label] = 0.68
        
        total = sum(probs.values())
        normalized_probs = {k: round(v / total, 4) for k, v in probs.items()}
        
        return {
            "predicted_label": best_label,
            "confidence": normalized_probs[best_label],
            "probabilities": normalized_probs
        }

predictor = ModelPredictor()
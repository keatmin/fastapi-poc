import ktrain
from typing import Dict, List


MODEL_DIR = "models/depression_ai"
learner_class = ["No depression", "Depression"]


class DepressionAI:
    def __init__(self):
        self.learner = ktrain.load_predictor(MODEL_DIR)
        
    def predict(self, text: str) -> Dict[str, str]:
        probability = self.learner.predict(text,return_proba=True)
        predicted_index = probability.argmax()
        return {"result": learner_class[predicted_index],
                "probability": probability[predicted_index]}

import ktrain
from ktrain import text as txt
import pickle
from typing import Dict, List

from tensorflow.keras.backend import set_session


class DepressionAI:
    def __init__(self):
        self.model, self.graph, self.sess, self.features = ktrain

    def preprocess_text(self, text: str) -> List[float]:
        vectorized_text = self.features.preprocess(text)
        return vectorized_text

    def predict(self, text: str) -> Dict[str]:
        vec_query = self.preprocess_text(text)
        with self.graph.as_default():
            set_session(self.sess)
            vec_cat = self.model.predict(vec_query)

            predicted_index = vec_cat[0].argmax(axis=0)
            return predicted_index

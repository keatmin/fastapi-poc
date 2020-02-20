from tensorflow.keras import models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.backend import get_session
from tensorflow.keras import backend as K
import pickle
import tensorflow as tf

MODEL_JSON = "models/model.json"
PICKLE_FILE = "models/model.preproc"
MODEL_WEIGHTS = "models/model.h5"


def load_files():
    with open(MODEL_JSON, "r") as json_file:
        loaded_model_json = json_file.read()

    features = pickle.load(PICKLE_FILE, "rb")

    sess = K.get_session()
    graph = tf.get_default_graph()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(MODEL_WEIGHTS)

    return loaded_model, graph, sess, features

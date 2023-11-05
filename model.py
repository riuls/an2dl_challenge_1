import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class Model:
    def __init__(self, path):
        self.model = load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X):
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)  # Shape [BS]

        return out
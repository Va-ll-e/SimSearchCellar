# Aeon requires python v.12
# pip install aeon[all]
from aeon.datasets import load_airline
import numpy as np
from aeon.anomaly_detection import STOMP

X = np.random.default_rng(42).random((10, 2), dtype=np.float64)
detector = STOMP(X, window_size=2)
detector.fit_predict(X, axis=0)


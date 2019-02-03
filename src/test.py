from sklearn.metrics import log_loss
import numpy as np
import shelve
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

ture_labels = np.array([-1, 0, 1, 2, 3, 4, 5])
ture_labels = np.eye(7)
good_probs = np.eye(7)
bad_probs = (np.ones((7, 7)) - good_probs) / 6.0
print("hello")
print('Good: ', log_loss(ture_labels, good_probs))
print('Bad: ', log_loss(ture_labels, bad_probs))



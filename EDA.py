import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

data = tfds.load('bridge_dataset',
                 data_dir="D:/bridge_data",
                 split='train')
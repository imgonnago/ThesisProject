import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import platform

def data_info():
    if platform.system() == 'Window':
        builder = tfds.builder_from_directory('D:/bridge_data/bridge_orig/1.0.0')
        data = builder.as_dataset(split='train')
        print(builder.info)
        return data
    elif platform.system() == 'Darwin':
        builder = tfds.builder_from_directory('/Volumes/YONGJAE/bridge_data/bridge_orig/1.0.0')
        data = builder.as_dataset(split='train')
        print(builder.info)
        return data

data = data_info()
for episode in data.take(1):
    steps = list(episode['steps'])
    print("에피소드 길이:", len(steps))

    step = steps[0]
    print("언어 지시문:", step['language_instruction'].numpy().decode('utf-8'))
    print("action:", step['action'].numpy())
    print("state:", step['observation']['state'].numpy())

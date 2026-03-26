import tensorflow_datasets as tfds
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import platform
from tqdm import tqdm

def data_load():
    print('=====data lodad=====')
    print('=====data info======')
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

data = data_load()

def show_image(cnt=3):
    global data
    image_map = {
        'image0': 'image_0',
        'image1': 'image_1',
        'image2': 'image_2',
        'image3': 'image_3',
    }

    chooes_image = input('보고싶은 이미지를 선택: ')
    for i,episode in enumerate(tqdm(data.take(cnt))):

        steps = list(episode['steps'])
        step = steps[0]

        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        img = step['observation'][image_map[chooes_image]].numpy()
        axes.imshow(img)
        axes.axis('off')
        plt.title(f"에피소드 {i}")
    plt.show()

def show_data(cnt=3):
    global data
    for i, episode in enumerate(tqdm(data.take(cnt))):
        steps = list(episode['steps'])
        print("에피소드 길이:", len(steps))

        step = steps[0]
        print("언어 지시문:", step['language_instruction'].numpy().decode('utf-8'))
        print("action:", step['action'].numpy())
        print("state:", step['observation']['state'].numpy())

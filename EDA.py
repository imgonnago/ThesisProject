import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

builder = tfds.builder_from_directory('D:/bridge_data/bridge_orig/1.0.0')
data = builder.as_dataset(split='train')
print(builder.info)

# 데이터 로드
data = builder.as_dataset(split='train')

# 에피소드 1개 확인
for episode in data.take(1):
    steps = list(episode['steps'])
    print("에피소드 길이:", len(steps))

    # 첫 번째 스텝 확인
    step = steps[0]
    print("언어 지시문:", step['language_instruction'].numpy().decode('utf-8'))
    print("action:", step['action'].numpy())
    print("state:", step['observation']['state'].numpy())
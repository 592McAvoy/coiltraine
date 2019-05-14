import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def draw(label, title):
    plt.subplots()
    plt.imshow(label, cmap='jet')
    plt.title(title)
    plt.colorbar()
    plt.show()

dirname = "test_label"
label_dir = os.listdir(dirname)
for label_file in label_dir:
    label = np.array(Image.open(dirname+"/"+label_file))
    print(np.unique(label))

#    draw(label, label_file)


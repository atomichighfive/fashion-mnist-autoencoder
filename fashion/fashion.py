import gzip
import numpy as np
import matplotlib.pyplot as plt

def read_data(images_path, labels_path, num_images = 10):
    image_file =  gzip.open(images_path,'r')
    label_file = gzip.open(labels_path,'r')
    image_size = 28

    image_file.read(16)
    buf = image_file.read(image_size * image_size * num_images)
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images = images.reshape(num_images, image_size, image_size, 1)

    label_file.read(8)
    buf = label_file.read(num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.uint8)
    #labels = labels.reshape(num_images)

    return images, labels

def plot_sample_labels(images, labels, label_dict = {}):
    fig = plt.figure(figsize=(10,10))
    kinds = sorted(np.unique(labels))
    cols = int(np.ceil(np.sqrt(len(kinds))))
    rows = int(np.ceil(len(kinds)/cols))

    for i, label in enumerate(kinds):
        index = np.random.choice(np.argwhere(labels == label).squeeze(),1)
        image = np.asarray(images[index]).squeeze()
        plt.subplot(rows, cols, i+1)
        plt.imshow(image, aspect='equal', cmap='pink')
        plt.title(label_dict[label] if label in label_dict.keys() else str(label))

def plot_image(image):
    plt.imshow(image.squeeze(), cmap='pink')

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_filters = 8
filter_size = 3
pool_size = 2


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def main():
    print(f'Tensorflow version: {tf.__version__}')
    v = sys.version_info
    print(f'Running under Python {v[0]}.{v[1]}.{v[2]}')
    print('Paths: ' + '\n'.join(sys.path))
    dirpath = os.getcwd()
    print(f'Current directory is {dirpath}')
    foldername = os.path.basename(dirpath)
    print(f'Directory name is {foldername}')
    print(f'Eager execution: {tf.executing_eagerly()}')
    tf_cuda_support = tf.test.is_built_with_cuda()
    gpu_available = len(tf.config.list_physical_devices('GPU'))

    print(f'Cuda support: {tf_cuda_support}')
    print(f'GPU available: {gpu_available}')

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    # Normalize the images.
    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5

    # Reshape the images.
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    model = keras.models.Sequential([
        keras.layers.Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=pool_size),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)
    predictions = model.predict(test_images)

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()


if __name__ == '__main__':
    main()

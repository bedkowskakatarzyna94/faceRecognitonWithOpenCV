import cv2
import numpy as np
import os
import re


def read_dataset_and_labels(directory_path):
    images = []
    labels = []
    for file in os.listdir(directory_path):
        if '.pgm' in file:
            image = cv2.imread(os.path.join(directory_path, file), 0)
            image = cv2.resize(image, (64, 64))
            images.append(image)
            label = re.findall('[0-9]+', file)
            labels.append(int(''.join(label)))
    return tuple([images, labels])


if __name__ == '__main__':

    model_name = 'lbp'

    training_data, training_labels = read_dataset_and_labels(
        "/Users/kasia/Desktop/A-Comparative-Study-of-PCA-and-LBP-for-Face-Recognition-main/training_data")
    test_data, test_labels = read_dataset_and_labels(
        "/Users/kasia/Desktop/A-Comparative-Study-of-PCA-and-LBP-for-Face-Recognition-main/test_data/without_filter")

    if model_name == 'lbp':
        model = cv2.face.LBPHFaceRecognizer_create()
    elif model_name == 'pca':
        model = cv2.face.EigenFaceRecognizer_create()
    else:
        raise AttributeError()

    model.train(training_data, np.asarray(training_labels))

    correct = 0
    for i in range(len(test_data)):
        prediction = model.predict(test_data[i])
        print(prediction)
        if prediction[0] == test_labels[i]:
            correct += 1

    print(f"{model_name} efficiency: " + str(correct / len(test_data)))

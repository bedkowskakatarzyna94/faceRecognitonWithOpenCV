import uuid

import cv2
import numpy as np
import os
import re


def save_result(file_name, text):
    file = open('/Users/kasia/Desktop/A-Comparative-Study-of-PCA-and-LBP-for-Face-Recognition-main/output_data' + file_name + '.txt', 'a')
    file.write(text + '\n')
    file.close()


def read_dataset_and_labels(directory_path):
    images = []
    labels = []
    for file in os.listdir(directory_path):
        if '.pgm' in file:  # if '.pgm' in file:    -depends on the file format
            if file is not None:
                image = cv2.imread(os.path.join(directory_path, file), 0)
                image = cv2.resize(image, (320, 243))
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

    filename = str(uuid.uuid4())

    correct = 0
    for i in range(len(test_data)):
        prediction = model.predict(test_data[i])
        print(prediction)
        text_line = str(prediction[0]) + ", " + str(prediction[1])
        save_result(filename, text_line)
        if prediction[0] == test_labels[i]:
            correct += 1

    print("samples: ", len(test_data), " ", "correct predictions: ", correct)
    print(f"{model_name} efficiency: " + str(correct / len(test_data)))

    save_result(filename, str("efficiency: " + str(correct / len(test_data))))


# -------------------------------------------------------------------------------------------------

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def detect_face(img):
    tmp = img.copy()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(tmp, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return tmp[y:y + w, x:x + h], faces[0]


subjects = ["", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15"]


def predict_img(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    id_confidence = model.predict(img)
    label_text = subjects[id_confidence[0]] + " " + str(id_confidence[1])
    if rect is not None:
        draw_rectangle(img, rect)
    draw_text(img, label_text, 5, 20)
    return img


for i in range(len(test_data)):
    prediction = model.predict(test_data[i])
    predicted_img1 = predict_img(cv2.resize(test_data[i], (320, 243)))
    title_win = str(prediction[0]) + "prediciton"
    cv2.imshow(title_win, cv2.resize(predicted_img1, (320, 243)))
    k = cv2.waitKey(0)
    if k == 27:
        break
    cv2.destroyAllWindows()

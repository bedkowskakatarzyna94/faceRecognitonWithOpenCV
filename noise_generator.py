import numpy as np
import cv2
import os
import re
import random
import string


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def add_noise(directory_path):
    char = 0
    for file in os.listdir(directory_path):
        if '.pgm' in file:
            image = cv2.imread(os.path.join(directory_path, file), 0)
            image = cv2.resize(image, (64, 64))
            noise_img = sp_noise(image, 0.06)
            label = re.findall('[0-9]+', file)
            char += 1
            filename = ''.join(label) + "noise_" + string.ascii_letters[char] + ".jpg"
            os.chdir("/Users/kasia/Desktop/A-Comparative-Study-of-PCA-and-LBP-for-Face-Recognition-main/test_data/with_noise_6")
            cv2.imwrite(filename, noise_img)


if __name__ == '__main__':
    add_noise("/Users/kasia/Desktop/A-Comparative-Study-of-PCA-and-LBP-for-Face-Recognition-main/test_data/without_filter")

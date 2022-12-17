# import os
# from skimage.io import imread
# from skimage.transform import resize
# import numpy as np
#
# from sklearn.decomposition import PCA
# from skimage.feature import local_binary_pattern
# from sklearn.model_selection import train_test_split
# from scipy.spatial import distance
# import matplotlib.pyplot as plt
#
#
# def read_data(path, height, width, for_pca=False):
#     labels, data = [], []
#     class_counter = 0
#
#     class_directories = os.listdir(path)
#     for class_directory in class_directories:
#         if (class_directory[0] == '.'):
#             continue
#         images = os.listdir(path + class_directory)
#         class_counter += 1
#         for image_name in images:
#             if (image_name.split('.')[-1] == 'pgm'):
#                 image_path = path + class_directory + '/' + image_name
#                 image = imread(image_path, 0)
#                 image = resize(image, (height, width), mode='constant', preserve_range=True)
#                 if (for_pca):
#                     image = image.reshape(-1)
#                 data.append(image)
#                 labels.append(class_counter)
#     data, labels = shuffle_data(data, labels)
#     data = np.array(data, dtype=np.float32) / 255.
#     labels = np.array(labels, dtype=np.uint8)
#     return data, labels
#
#
# def read_single_image(path, height, width):
#     image = imread(path, 0)
#     image = resize(image, (height, width), mode='constant', preserve_range=True)
#     image = np.array(image, dtype=np.float32) / 255.
#     return image
#
#
# def shuffle_data(data, labels):
#     len_of_data = len(data)
#     for i in range(len_of_data // 2):
#         x = np.random.randint(len_of_data)
#         temp = data[x]
#         data[x] = data[i]
#         data[i] = temp
#
#         temp = labels[x]
#         labels[x] = labels[i]
#         labels[i] = temp
#     return data, labels
#
#
# def experiment_with_pca(X_train, X_test, y_train, y_test):
#     n_pca_components = X_train.shape[0]
#
#     # extract features with pca and get eigen values
#     pca = PCA(n_components=n_pca_components)
#     pca = pca.fit(X_train)
#     # eig_faces = pca.components_.reshape(n_pca_components, height, width)
#     features = pca.transform(X_train)
#     test_features = pca.transform(X_test)
#
#     # classification with euclidean metric
#     # we are looking for distance to the nearest neighbor
#     preds = []
#     for test_feature in test_features:
#         min_dist = 10000
#         for (feature, y) in zip(features, y_train):
#             dist = abs(distance.euclidean(feature, test_feature))
#             if (min_dist > dist):
#                 min_dist = dist
#                 min_y = y
#         preds.append(min_y)
#     return preds, y_test
#
#
# def get_hist_from_lbph(data):
#     # reference:
#     # pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
#     eps = 1e-7
#     radius = 4
#     n_points = 8 * radius
#     lbp = local_binary_pattern(data, n_points, radius, 'uniform')
#     (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
#     hist = np.array(hist, dtype=np.float32)
#     return hist
#
#
# def chi2_distance(histA, histB, eps=1e-10):
#     # reference:
#     # pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
#     # compute the chi-squared distance
#     d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
#     return d
#
#
# def experiment_with_lbp(X_train, X_test, y_train, y_test):
#     features, test_features = [], []
#     for X_train_sample in X_train:
#         features.append(get_hist_from_lbph(X_train_sample))
#     for X_test_sample in X_test:
#         test_features.append(get_hist_from_lbph(X_test_sample))
#
#     # classification with euclidean metric
#     # we are looking for distance to the nearest neighbor
#     preds = []
#     for test_feature in test_features:
#         min_dist = 10000
#         for (feature, y) in zip(features, y_train):
#             dist = abs(chi2_distance(feature, test_feature))
#             if (min_dist > dist):
#                 min_dist = dist
#                 min_y = y
#         preds.append(min_y)
#     return preds, y_test
#
#
# if (__name__ == '__main__'):
#     seed = 0
#     height, width = 512, 512
#     dataset_path = 'CroppedYale/'
#
#     # read data
#     # data, labels = read_data(dataset_path, height, width, for_pca=True)
#     # split (feature extracted) data
#     # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=seed)
#
#     # experiment with pca
#     # pca_preds, pca_gt = experiment_with_pca(X_train, X_test, y_train, y_test)
#
#     # read data
#     data, labels = read_data(dataset_path, height, width)
#     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=seed)
#
#     # experiment with lbp
#     lbp_preds, lbp_gt = experiment_with_lbp(X_train, X_test, y_train, y_test)
#
#     from sklearn.metrics import accuracy_score
#
#     print(accuracy_score(lbp_gt, lbp_preds))

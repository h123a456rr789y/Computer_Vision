import cv2
import os
import glob
import numpy as np
import math
import faiss
import cyvlfeat as vlfeat

sift = cv2.SIFT_create(nOctaveLayers = 5)

def dataloader(filepath):
    labels = []
    image = []
    for label in os.listdir(filepath):
        imgs = glob.glob(os.path.join(filepath, label)+"/*.jpg")
        for img in imgs:
            img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img,(32,32))        
            image.append(img)
            labels.append(label)
    return image, labels

def preprocessing(image):
    #sift = cv2.SIFT_create()
    
    #sift = cv2.SIFT_create(nfeatures = 2000, nOctaveLayers = 5, contrastThreshold = 0.5, edgeThreshold = 0.5, sigma = 0.5, octave = 2000)
    number = 0
    
    for i in range(len(image)):
        #kp, des = sift.detectAndCompute(image[i], None)
        kp, des = vlfeat.sift.dsift(image[i], fast=False, step=11)
        if len(kp) > 0:
            if number == 0:
                feature = des
                number += 1
            else :
                feature = np.vstack((feature, des)).astype('float32')  
    return feature
    
train_image, train_label = dataloader('hw5_data/train/')
test_image, test_label = dataloader('hw5_data/test/')
train_feature = preprocessing(train_image)
test_feature = preprocessing(test_image)

print(train_feature.shape)



print("Start K-means")
kmeans_k = 100
niter = 50
nredo = 1
verbose = True
d = train_feature.shape[1]
kmeans = faiss.Kmeans(d, kmeans_k, niter=niter, verbose=verbose)
kmeans.train(train_feature)
Cluster_Mean = kmeans.centroids
_ , belong = kmeans.index.search(train_feature, 1)


def IMG_content(img, k):
    content = np.zeros((len(img), k))
    label = []
    for i in range(len(img)):
        kp, des = vlfeat.sift.dsift(img[i], fast=False, step=11)
        des = np.array(des).astype('float32')
        if len(kp) > 0:
            _, belong = kmeans.index.search(des, 1)
            number , counts = np.unique(belong, return_counts = True)
            content[i, number] = counts
    return content


train_content = IMG_content(train_image, kmeans_k)
test_content = IMG_content(test_image, kmeans_k)


def knn(train_set, train_label, test_set, k=10):
    pred_label = []
    for test_img in test_set:
        dis = (abs(train_set-test_img)).sum(axis=1)
        # dis = ((train_set - test_img)**2).sum(axis=1)**0.5
        sort_index = dis.argsort()[:k]
        knn_labels = []
        for i in sort_index:
            knn_labels.append(train_label[i])
        values, counts = np.unique(knn_labels, return_counts=True)
        ind = np.argmax(counts)
        pred_label.append(values[ind])
    return np.array(pred_label)
      
def cal_accuracy(pred_label, test_label):
    return np.sum(pred_label == test_label) / len(test_label)        
        
def testing(train_content, train_label, test_content):
    pred_label = knn(train_content, train_label, test_content, k = knn_k)
    return pred_label

knn_k = 15
pred_label = testing(train_content, train_label, test_content)
accuracy = cal_accuracy(pred_label, test_label)
print(f"Accuracy is {accuracy:.03f}")



kmeans_k_array = [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 400]
knn_k_array = [10, 15, 20, 25, 30, 35, 40]

#kmeans_k_array = [150]
#knn_k_array = [15, 20]

niter = 100
nredo = 1
verbose = True
d = train_feature.shape[1]
for kmeans_k in kmeans_k_array:
    for knn_k in knn_k_array:
        print("Start K-means")
        kmeans = faiss.Kmeans(d, kmeans_k, niter=niter, verbose=verbose)
        kmeans.train(train_feature)
        Cluster_Mean = kmeans.centroids
        train_content = IMG_content(train_image, kmeans_k)
        test_content = IMG_content(test_image, kmeans_k)
        pred_label = testing(train_content, train_label, test_content)
        accuracy = cal_accuracy(pred_label, test_label)
        print(f"kmeans_k is {kmeans_k}\t knn_k is {knn_k}\t accurayc is {accuracy:.03f}")
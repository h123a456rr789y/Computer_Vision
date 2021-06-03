import cv2
import os
import glob
import numpy as np
import faiss
from libsvm.svmutil import *
import cyvlfeat as vlfeat


def dataloader(filepath):
    labels = []
    image = []
    i=0
    for label in os.listdir(filepath):
        imgs = glob.glob(os.path.join(filepath, label)+"/*.jpg")
        for img in imgs:
            img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img,(16,16))        
            image.append(img)
            labels.append(i)
        i+=1
    return image, labels

def preprocessing(image):
    number = 0
    for i in range(len(image)):
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
train_feature =preprocessing(train_image)
test_feature = preprocessing(test_image)

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


print("Start K-means+ SVM")

kmeans_k_array = [100, 200, 400, 800, 1000, 1200, 1500]

for kmeans_k in kmeans_k_array:
    niter = 100
    nredo = 1
    verbose = False
    d = train_feature.shape[1]
    kmeans = faiss.Kmeans(d, kmeans_k, niter=niter, verbose=verbose)
    kmeans.train(train_feature)
    Cluster_Mean = kmeans.centroids
#         print(Cluster_Mean.shape)
    _ , belong = kmeans.index.search(train_feature, 1)
#         print(belong.shape)
#         print(belong)



    train_content = IMG_content(train_image, kmeans_k)
    test_content = IMG_content(test_image, kmeans_k)
#         print(train_content.shape)

    ##SVM

    para_str= "-c 10000 -e 0.0001 -t 0"
    prob = svm_problem(train_label,train_content)
    param = svm_parameter(para_str)
    model = svm_train(prob,param)   
    print(f"kmeans_k is {kmeans_k}\t")
    p_label, p_acc, p_val = svm_predict(test_label, test_content, model)
        
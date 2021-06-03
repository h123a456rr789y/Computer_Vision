
# coding: utf-8

# In[2]:


import cv2
import os
import glob
import numpy as np


# In[3]:


def data_preprocessing(filepath):
    labels = []
    tiny_imgs = []
    for label in os.listdir(filepath):
        imgs = glob.glob(os.path.join(filepath, label)+"/*.jpg")
        for img in imgs:
            img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(16,16))
            tiny_imgs.append(img)
            labels.append(label)
    return tiny_imgs, labels


# In[4]:


def cal_accuracy(pred_label, test_label):
    return np.sum(pred_label == test_label) / len(test_label)


# In[16]:


def knn(train_set, train_label, test_set, k=10):
    pred_label = []
    for test_img in test_set:
#         dis = (abs(train_set-test_img)).sum(axis=1)
        dis = ((train_set - test_img)**2).sum(axis=1)**0.5
        sort_index = dis.argsort()[:k]
        knn_labels = []
        for i in sort_index:
            knn_labels.append(train_label[i])
        values, counts = np.unique(knn_labels, return_counts=True)
        ind = np.argmax(counts)
        pred_label.append(values[ind])
    return pred_label


# ### 1.Tiny images representation + nearest neighbor classifier

# In[13]:


train_set, train_label = data_preprocessing('hw5_data/train/')
train_set = np.array(train_set)
train_set = train_set.reshape(train_set.shape[0], -1)
test_set, test_label = data_preprocessing('hw5_data/test/')
test_set = np.array(test_set)
test_set = test_set.reshape(test_set.shape[0], -1)
print(train_set.shape)
print(test_set.shape)
best_k = 0
which_k = 0
for k in range(3,50):
    pred_label = knn(train_set, train_label, test_set, k=k)
    a = cal_accuracy(np.array(pred_label), np.array(test_label))
#     print(k, a)
    if a > best_k:
        best_k = a
        which_k =k
print(best_k)
print(which_k)


# In[17]:


train_set, train_label = data_preprocessing('hw5_data/train/')
train_set = np.array(train_set)
train_set = train_set.reshape(train_set.shape[0], -1)
test_set, test_label = data_preprocessing('hw5_data/test/')
test_set = np.array(test_set)
test_set = test_set.reshape(test_set.shape[0], -1)
print(train_set.shape)
print(test_set.shape)
best_k = 0
which_k = 0
for k in range(3,50):
    pred_label = knn(train_set, train_label, test_set, k=k)
    a = cal_accuracy(np.array(pred_label), np.array(test_label))
#     print(k, a)
    if a > best_k:
        best_k = a
        which_k =k
print(best_k)
print(which_k)


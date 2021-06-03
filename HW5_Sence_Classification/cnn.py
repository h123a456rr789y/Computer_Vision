import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time
import os
import PIL.Image as Image
from IPython.display import display

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))


dataset_dir = "./hw5_data/"
batch_size = 30
train_tfms = transforms.Compose([transforms.Resize((250, 250)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                      (0.229, 0.224, 0.225))])
test_tfms = transforms.Compose([transforms.Resize((250, 250)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                      (0.229, 0.224, 0.225))])

dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train",
                                           transform=train_tfms)

train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=False)
dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir+"test",
                                            transform=test_tfms)
testloader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size,
                                         shuffle=False)

def train_model(model, criterion, optimizer, scheduler, n_epochs=10):
    losses = []
    accuracies = []
    val_accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()
    for epoch in range(n_epochs):
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs and assign them to cuda
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()
        epoch_duration = time.time()-since
        epoch_loss = running_loss/len(trainloader)
        epoch_acc = 100/batch_size*running_correct/len(trainloader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        # switch the model to eval mode to evaluate on test data
        model.eval()
        val_acc = eval_model(model, valloader,"validation")
        val_accuracies.append(val_acc)
        # re-set the model to train mode after validating
        model.train()
        scheduler.step(val_acc)
        since = time.time()
        
        model.eval()
        test_acc = eval_model(model, testloader,"test")
        test_accuracies.append(test_acc)
    
    print('Finished Training')
#     epoch = [e for e in range(1,n_epochs+1)]
#     plt.plot(epoch, accuracies, label="train")
#     plt.plot(epoch, test_accuracies, label="test")
#     plt.title("ResNet50")
#     plt.xlabel("epoch")
#     plt.ytitle("accuracy")
#     plt.show()
    return model, losses, accuracies, val_accuracies,test_accuracies

def eval_model(model,testloader, mode):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100.0 * correct / total
    print(f'Accuracy of the network on the {mode} images: %d %%' % (
        test_acc))
    return test_acc

# load pretrain model
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
# replace the last fc layer with an untrained one (requires grad by default)
model_ft.fc = nn.Linear(num_ftrs, 15)
model_ft = model_ft.to(device)
# loss function and optimzer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold=0.9)

model_ft, training_losses, training_accs, acc_accs,test_accs = train_model(model_ft, criterion, optimizer, lrscheduler, n_epochs=50)

epoch = [e for e in range(1,51)]
plt.plot(epoch, training_accs, label="train",marker="x")
plt.plot(epoch, test_accs, label="test",marker="x")
plt.title("ResNet50")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.savefig("resnet.jpg")
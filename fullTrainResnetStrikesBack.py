import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from fastcore.all import *
from fastai.vision.all import *
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import ImageFolder
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from datetime import datetime
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, input_planes, output_planes,stride,identityFlag):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, output_planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn1 =  nn.BatchNorm2d(output_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_planes, output_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 =  nn.BatchNorm2d(output_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.downsample = None
        if identityFlag:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_planes, output_planes, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(output_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Sequential):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                nn.Sequential(
                    BasicBlock(input_planes=64, output_planes=64,stride=1,identityFlag=False),
                    BasicBlock(input_planes=64, output_planes=64,stride=1,identityFlag=False)
                ),
                nn.Sequential(
                    BasicBlock(input_planes=64, output_planes=128, stride=2, identityFlag=True),
                    BasicBlock(input_planes=128, output_planes=128, stride=1, identityFlag=False)
                ),
                nn.Sequential(
                    BasicBlock(input_planes=128, output_planes=256, stride=2, identityFlag=True),
                    BasicBlock(input_planes=256, output_planes=256, stride=1, identityFlag=False)
                ),
                nn.Sequential(
                    BasicBlock(input_planes=256, output_planes=512, stride=2, identityFlag=True),
                    BasicBlock(input_planes=512, output_planes=512, stride=1, identityFlag=False)
                )
            )

        self.layer2 = nn.Sequential(
                AdaptiveConcatPool2d(1),
                Flatten(full=False),
                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Dropout(p=0.25, inplace=False),
                nn.Linear(in_features=1024, out_features=512, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=512, out_features=6, bias=False)
            )
        def forward(self,x):
            out = self.layer1(x)
            out = self.layer2(out)
            return out

seed_value = 60
torch.manual_seed(seed_value)
np.random.seed(seed_value)


file_path_to_save_stats = "stats/statsFullTrainResnetStrikesBack.csv"
file_path_to_save_model = "models/mlprojectFullTrainResnetStrikesBack.pth"
dataset_root = 'C:/Users/sriha/NEU/clozy cloth types/garmentStructuredData'
model = ResNet18()
if os.path.exists(file_path_to_save_model):
    model.load_state_dict(torch.load(file_path_to_save_model))
    print("Model weights found loaded from path :",file_path_to_save_model)
else:

    print("No pretrained weights found")
lrn_rate = 0.001
num_epochs = 50
print("Learning rate : ",lrn_rate)
print("Epochs : ",num_epochs)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lrn_rate, weight_decay = 0.1)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0001)


num_classes = 6
transform = transforms.Compose([
    Resize(192, method='squish'),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])
custom_dataset = ImageFolder(root=dataset_root, transform=transform)
class_names = custom_dataset.classes
print("Class names : ",class_names)


train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

torch.manual_seed(torch.initial_seed())


for param in model.parameters():
    param.requires_grad = True

def validate_on_test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))
    print('Class names (labels):', class_names)

def compute_test_accuracy_loss():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss = test_loss + loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.detach().numpy())
            y_pred.extend(predicted.detach().numpy())

    print('Accuracy on test set: %d %%' % (100 * correct / total))
    return 100 * (correct / total), test_loss, y_true, y_pred

def compute_precision_recall(labels, predictions):
    precision = precision_score(labels, predictions,average='macro',zero_division=1)
    recall = recall_score(labels, predictions,average='macro',zero_division=1)
    return precision, recall

def compute_f1_score(y_true,y_pred):
    f1 = f1_score(y_true, y_pred, average='macro',zero_division=1)  # 'macro' computes the F1 score for each class independently and then takes the average
    return f1

batch_size_stack = []
test_loss_stack = []
train_loss_stack = []
test_accuracy_stack = []
train_accuracy_stack = []
time_epoch = []
precision_train = []
precision_test = []
recall_train = []
recall_test  = []
f1_score_train = []
f1_score_test = []
learning_rate=[]
timestamps = []
iters = len(train_loader)

for epoch in range(num_epochs):
    start_epoch_time = time.time()
    epoch_loss_training = 0
    correct = 0
    total = 0
    model.train()
    running_loss = 0.0

    y_pred = []
    y_true = []

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

        y_true.extend(labels.detach().numpy())
        y_pred.extend(predicted.detach().numpy())
        scheduler.step(epoch + i / iters)
        running_loss += loss.item()
        epoch_loss_training = epoch_loss_training + loss.item()
        # if i % 5 == 4:    # Print every 10 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 5))
        #     running_loss = 0.0
    print ("[{epoch}] loss : {loss} Lr : {lr}".format(epoch=epoch,loss=epoch_loss_training,lr=scheduler.get_last_lr()))
    end_training_time = time.time()
    acc, test_loss, test_y_true, test_y_pred = compute_test_accuracy_loss()
    precision, recall = compute_precision_recall(y_true,y_pred)
    test_precision, test_recall = compute_precision_recall(test_y_true,test_y_pred)
    f1_train = compute_f1_score(y_true,y_pred)
    f1_test = compute_f1_score(test_y_true,test_y_pred)

    time_epoch.append(round(end_training_time-start_epoch_time,4))

    recall_test.append(round(test_recall,4))
    precision_test.append(round(test_precision,4))

    precision_train.append(round(precision,4))
    recall_train.append(round(recall,4))

    f1_score_train.append(round(f1_train,4))
    f1_score_test.append(round(f1_test,4))

    test_accuracy_stack.append(round(acc,4))
    train_accuracy_stack.append(round(100 * (correct / total),4))

    test_loss_stack.append(round(test_loss,4))
    train_loss_stack.append(round(epoch_loss_training,4))
    learning_rate.append(round(scheduler.get_last_lr()[0],8))
    t = datetime.now()
    formatted_date = t.strftime('%Y-%m-%d %H:%M:%S')
    timestamps.append(formatted_date)
    batch_size_stack.append(batch_size)

if not os.path.exists(file_path_to_save_stats):
    df = pd.DataFrame(
        data = {
            'timestamps':[],
            'time_epoch':[],
            'batch_size':[],
            'learning_rate':[],
            'f1_score_test':[],
            'f1_score_train':[],
            'recall_test':[],
            'recall_train':[],
            'precision_test':[],
            'precision_train':[],
            'train_accuracy':[],
            'test_accuracy':[],
            'train_loss':[],
            'test_loss':[]
        }
    )
    df.to_csv(file_path_to_save_stats,index=False)

df = pd.DataFrame(
    data = {
        'timestamps':timestamps,
        'time_epoch':time_epoch,
        'batch_size':batch_size_stack,
        'learning_rate':learning_rate,
        'f1_score_test':f1_score_test,
        'f1_score_train':f1_score_train,
        'recall_test':recall_test,
        'recall_train':recall_train,
        'precision_test':precision_test,
        'precision_train':precision_train,
        'train_accuracy':train_accuracy_stack,
        'test_accuracy':test_accuracy_stack,
        'train_loss':train_loss_stack,
        'test_loss':test_loss_stack
    }
)
df.to_csv(file_path_to_save_stats,index=False,mode='a',header=False)
torch.save(model.state_dict(), file_path_to_save_model)
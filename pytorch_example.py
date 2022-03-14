#!/usr/bin/env python

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torchsummary

import os
import cv2
import re
import numpy as np
import math
import time
from datetime import timedelta

# pytorch의 시작과 같은 Code
# 그래픽카드를 사용할 수 있으면 사용하고 그렇지 않으면 CPU를 사용한다
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=(0, 1), padding_mode='circular', bias=False)

BN_MOMENTUM = 0.1
class BasicBlock(nn.Module): # 모델 정의, __init__, forward를 정의한다
    expansion = 1

    # __init__에 선언된 NN(Neural Network)은 자동으로 변수 초기화가 실행된다 그렇기 때문에 모든 NN은 __init__에서 선언되어야 한다
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__() # BasicBlock의 부모(parent)인 nn.Module에 선언된 __init__을 실행한다. 필수적임

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # Forward Propagation
        # 이 함수 이름 forward는 반드시 지켜서 선언해야한다.
        residual = x

        out = F.pad(x, (0, 0, 1, 1))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, (0, 0, 1, 1))
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MyCNN(nn.Module): # 모델 정의 My Convolutional Neural Network
    def __init__(self):
        super(MyCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=(0, 4), padding_mode='circular', bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        )
        self.layer2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.block64 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        )

        self.block128 = nn.Sequential(
            BasicBlock(64, 128, 2, self.downsample1),
            BasicBlock(128, 128, 1)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )


        self.block256 = nn.Sequential(
            BasicBlock(128, 256, 2, self.downsample2),
            BasicBlock(256, 256, 1)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM)
        )


        self.block512 = nn.Sequential(
            BasicBlock(256, 512, 2, self.downsample3),
            BasicBlock(512, 512, 1)
        )

        ## 여기까진 resnet-18과 비슷
        self.layer3 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        self.layer4 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        self.layer5 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.layer6 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size() # [N, C, H, W], H,W,C = 362, 1612, 3 or 512, 1612, 3
        h, w = h // 4, w // 4
        x = F.pad(x, (0, 0, 4, 4))
        x = self.layer1(x) # 91.5, 404 or 129, 404
        x = self.layer2(x)
        skip1 = self.block64(x) #  or 128, 403
        skip2 = self.block128(skip1) #  or 65, 202
        skip3 = self.block256(skip2) #  or 33, 101.5
        skip4 = self.block512(skip3) #  or 17, 51

        # out1 [N, 512, 66, 102]
        out1 = F.interpolate(skip4, scale_factor=2, mode='bilinear', align_corners=True)
        skip3 = F.pad(skip3, pad=(1, 0, 0, 0), mode='circular')
        concat_out1 = torch.cat((out1, skip3), dim=1) # concat_out1 = [N, 768, 66, 102], C = 512 + 256

        # out2 [N, 256, 132, 204]
        out2 = F.interpolate(self.layer3(concat_out1), scale_factor=2, mode='bilinear', align_corners=True)
        skip2 = F.pad(skip2, pad=(1, 1, 0, 0), mode='circular')
        concat_out2 = torch.cat((out2, skip2), dim=1) # concat_out2 = [N, 384, 132, 204], C = 256 + 128
        
        # out3 [N, 128, 264, 408]
        out3 = F.interpolate(self.layer4(concat_out2), scale_factor=2, mode='bilinear', align_corners=True)
        skip1 = F.pad(skip1, pad=(3, 2, 0, 0), mode='circular')
        concat_out3 = torch.cat((out3, skip1), dim=1) # concat_out3 [N, 192, 264, 408], C = 128 + 64

        # out4 [N, 64, 264, 408]
        output = F.interpolate(self.layer5(concat_out3), size=(h, w), mode='bilinear', align_corners=True) # output [N, 64, 512, 1612]

        output = self.layer6(output) # output [N, 1, 512, 1612]

        #verteices = torch.transpose(torch.squeeze((output >= 1.0).nonzero(as_tuple=False)), 0, -1)
        output = self.sigmoid(output)
        return output


class MyDataset1(Dataset): # 데이터 크기가 별로 안크고 메모리가 넉넉하고 코드 짜기 귀찮을 때 쓰는 방법
    def __init__(self, x, y):
        super(MyDataset1, self).__init__()
        self.x = x # 그냥 다 때려 넣는다
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])

class MyDataset2(Dataset): # 데이터 크기가 매우 커서 메모리가 감당을 못할 때, 디렉토리만 가져오고 __getitem__시 batch크기 만큼만 불러온다
    def __init__(self, x_path, y_path):
        super(MyDataset2, self).__init__()
        self.x_file_names = os.listdir(x_path) # 데이터 대신 데이터의 directory를 가져온다
        self.x_file_names.sort()
        self.x_file_list = [os.path.join(x_path, filename) for filename in self.x_file_names]

        self.y_file_names = os.listdir(y_path)
        self.y_file_names.sort()
        self.y_file_list = [os.path.join(y_path, filename) for filename in self.y2_file_names]

    def __len__(self):
        return len(self.x_file_list)

    def __getitem__(self, idx):
        x = np.transpose(cv2.imread(self.x_file_list[idx], cv2.IMREAD_COLOR), (2, 0, 1)) # 필요할때만 directory의 데이터를 가져온다
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(do_somthing(self.y_file_list[idx], self.y_file_list[idx]))
        return x, y # return값은 반드시 tensor type이어야 한다


if __name__ == '__main__':
    image_data_path = 'original_img'
    box3D_data_path = '3D_bounding_box'

    # image_data_path에 있는 파일의 개수
    dataset_len = len(os.listdir(image_data_path))

    # MyDataset2를 train set과 validation set으로 나눈다.
    # train set은 9할, validation set은 1할
    train_dataset, val_dataset = random_split(MyDataset2(image_data_path, box3D_data_path), [round(dataset_len * 0.9), round(dataset_len * 0.1)])

    # 각각의 dataloader를 만든다
    train_dataloader = DataLoader(train_dataset, batch_size=12, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=12, num_workers=0)

    # custom loss function 정의
    # loss function은 반드시 type이 tensor인 스칼라 값을 return 해야 한다
    def my_loss_function(X, y):
        return torch.sum(torch.square(100 * (y - X))) / len(y[y>0])

    NEW_MODEL = False

    if NEW_MODEL:
        model = MyCNN() # 모델 생성
        print('Training New Model')
    else:
        model = torch.load('best_model1.pt')
        print('load model')

    # 모델을 cuda로 넘겨준다
    model.to(device)

    # torchsummar, 모델의 구조를 요약해서 보여준다
    torchsummary.summary(model, (3, 512, 1612), batch_size=12, device=device)

    # optimizer로 Adam을 사용한다
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch = 100
    preval_loss, val_loss = 0.0, 0.0
    total_time, epoch_time, batch_time = time.time(), 0.0, 0.0
    MSE_funcion = nn.MSELoss()
    for i in range(epoch):
        epoch_time = time.time()
        print('epoch: {}'.format(i+1))

        model.to(device)
        model.train()
        batch_time = time.time()
        for batch, (X, Y) in enumerate(train_dataloader):
            X, Y = X.to(device), Y.to(device) # data를 cuda device에 넘겨줘야한다

            pred = model(X)
            loss = my_loss_function(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                batch_time = time.time() - batch_time
                print(f"loss: {loss:>7f}  [{current:>5d}/{round(dataset_len * 0.9):>5d}] --- time: {timedelta(seconds=round(batch_time))}")
                batch_time = time.time()
        print('train epoch done')

        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # valid시 무조건 넣는 코드
            for batch, (X, Y) in enumerate(val_dataloader):
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                val_loss += my_loss_function(pred, Y).item()
            if i == 0 or preval_loss > val_loss:
                torch.save(model, 'best_model2.pt')
                preval_loss = val_loss
                print(f'val_loss: {val_loss} --- val_loss decreased, best model saved.')
            else:
                print(f'val_loss: {val_loss} --- model not saved')
        epoch_time = time.time() - epoch_time
        print(f'time spent {timedelta(seconds=round(epoch_time))} per epoch')

    print('\n')
    print(f'total learning time: {timedelta(seconds=round(time.time() - total_time))}')

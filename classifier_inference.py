from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import utils

class opt():
    def __init__(self):
        self.testimg='sao.jpg'
        self.workers=2
        self.batchSize=4 #'input batch size') 배치사이즈
        self.imageSize=224 #the height / width of the input image to network') 이미지 사이즈 (정방)
        self.lr=0.0002 # 러닝레이트
        self.m=0.9
        #self.net='' # 트레이닝 처음 시작할때 이 주석을 해제함
        self.net='model/net_1.pth' # 트레이닝을 이어서 할때 이 주석을 해제함
        self.cuda=True # 쿠다 옵션 (CPU 트레이닝 일시에는 False)
        
opt = opt() # opt 클래스 오브젝트 생성

#트랜스폼 선언
ctransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage('RGB'),
    transforms.ToTensor()
])

import torchvision.models as models

# 모델 불러오기
cmodel = models.vgg16()
if opt.cuda:
    cmodel.cuda()

resume_epoch=0
if opt.net != '':
    cmodel.load_state_dict(torch.load(opt.net,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.net)['epoch']

#single
cimg = utils.load_image(opt.testimgpath, opt.cimageSize)
cimg = ctransform(cimg)
cinputs = torch.FloatTensor(1, 3, opt.cimageSize, opt.cimageSize)
if torch.cuda.is_available():
    cinputs = cinputs.cuda()
cinputs = Variable(inputs)
cinputs.data[0:] = img
outputs = model(cinputs)
predictions = outputs.max(1, keepdim=True)[1]
print("inference's predictions {}".format(predictions))
classes = ['makoto', 'sao', 'shinchan', 'zibri']
print(classes[predictions[0].data[0]])
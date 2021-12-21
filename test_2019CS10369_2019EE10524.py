import sys
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from IPython.display import Image
import cv2
from sklearn import preprocessing
import argparse
import math
from efficientnet_pytorch import EfficientNet



le = preprocessing.LabelEncoder()


Args=sys.argv
path_to_weights=os.path.join(Args[1],"model.pth")
path_to_testin=Args[2]
path_to_testout=Args[3]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available()) 



class ImageDataSet(Dataset):
    def __init__(self,csv_file,root_dir,transform,isTrain):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir  
        self.transform = transform
        self.isTrain = isTrain
        if(isTrain):
            labels = self.annotations.iloc[:,1]
            labels = le.fit_transform(labels)
            self.annotations['category'] = labels
        else:
            self.annotations['category'] = -1
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):
        img_path = self.annotations.iloc[index,0]        
        image = io.imread(img_path)      
        labels = self.annotations.iloc[index,1]
        y_label = torch.tensor(labels)
        if self.transform:
            image = self.transform(image)
        return (image,y_label)




transform = torchvision.transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((220,220)),
    transforms.RandomHorizontalFlip(p=0.5), # randomly flip and rotate
    torchvision.transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(0.3,0.4,0.4,0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
    ])


transform_tests = torchvision.transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((220,220)),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235))
    ])


class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network,self).__init__()
        self.effnet=EfficientNet.from_pretrained('efficientnet-b4')
        for param in self.effnet.parameters():
            param.required_grad = False
        self.l1=nn.Linear(1000,256)
        self.l2=nn.Dropout(0.75)
        self.l3=nn.Linear(256,19)
        self.l4=nn.ReLU()
        
    def forward(self,input):
        x=self.effnet(input)
        x=x.view(x.size(0),-1)
        x=self.l3(self.l2(self.l4(self.l1(x))))
        return x



batch_size=32
dataset = ImageDataSet(csv_file=path_to_testin,root_dir ="", transform = transform_tests,isTrain=False)
test_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)




model=Neural_Network().to(device)
model.load_state_dict(torch.load(path_to_weights))
model.eval()
predicted_arr=[]


with torch.no_grad():
      for i, (images,labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
      
        predicted_arr += list(predicted.cpu().numpy())

test_data =path_to_testin
P=pd.read_csv(test_data).to_numpy()

Labels=['Ardhachakrasana','Garudasana','Gorakshasana','Katichakrasana',
 'Natarajasana','Natavarasana','Naukasana','Padahastasana',
 'ParivrittaTrikonasana','Pranamasana','Santolanasana','Still','Tadasana',
 'Trikonasana','TriyakTadasana','Tuladandasana','Utkatasana',
 'Virabhadrasana','Vrikshasana']

le.fit(Labels)
fname=path_to_testout
L=le.inverse_transform(predicted_arr)
f=open(fname,"w")
s=str("name")+","+str("category")+"\n"
f.write(s)
for i in range(0,len(L)-1):
  s=str(P[i][0])+","+str(L[i])+"\n"
  f.write(s)
f.close()



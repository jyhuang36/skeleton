import os
import numpy as np
import scipy.misc
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Deep Skeleton')
parser.add_argument('-epochs', default=6, type=int, help='number of epochs')
parser.add_argument('-itersize', default=10, type=int, help='iteration size')
parser.add_argument('-printfreq', default=100, type=int, help='printing frequency')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-decay', default=0.0002, type=float, help='weight decay')
parser.add_argument('-mode', default='cpu', type=str, help='mode')
parser.add_argument('-gpuid', default=0, type=int, help='gpu id')
parser.add_argument('-train', default=False, action='store_true')
parser.add_argument('-visualize', default=False, action='store_true')
parser.add_argument('-test', default=True, action='store_true')



class Skeleton(nn.Module):
    def __init__(self):
        super(Skeleton, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),               
        )        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),               
        )               
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),   
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),              
        )                        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),                
        )                               
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),                 
        )
        self.dsn2 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.dsn3 = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear")
        )
        self.dsn4 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=1),
            nn.Upsample(scale_factor=8, mode="bilinear")
        )
        self.dsn5 = nn.Sequential(
            nn.Conv2d(512, 5, kernel_size=1),
            nn.Upsample(scale_factor=16, mode="bilinear")
        )
        
        self.fusion1 = nn.Conv2d(4, 1, kernel_size=1)
        self.fusion2 = nn.Conv2d(4, 1, kernel_size=1)
        self.fusion3 = nn.Conv2d(3, 1, kernel_size=1)
        self.fusion4 = nn.Conv2d(2, 1, kernel_size=1)
        self.fusion5 = nn.Conv2d(1, 1, kernel_size=1)
        
        self.init_weights()
     
    def forward(self, x):
        x_ = self.conv1(x)

        x_ = self.conv2(x_)
        y2 = self.dsn2(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        x_ = self.conv3(x_)
        y3 = self.dsn3(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        x_ = self.conv4(x_)
        y4 = self.dsn4(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        x_ = self.conv5(x_)
        y5 = self.dsn5(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        yf1 = self.fusion1(torch.cat((y2[:,0:1,:,:], y3[:,0:1,:,:], y4[:,0:1,:,:], y5[:,0:1,:,:]), 1))
        yf2 = self.fusion2(torch.cat((y2[:,1:2,:,:], y3[:,1:2,:,:], y4[:,1:2,:,:], y5[:,1:2,:,:]), 1))
        yf3 = self.fusion3(torch.cat((y3[:,2:3,:,:], y4[:,2:3,:,:], y5[:,2:3,:,:]), 1))
        yf4 = self.fusion4(torch.cat((y4[:,3:4,:,:], y5[:,3:4,:,:]), 1))
        yf5 = self.fusion5(y5[:,4:5,:,:])
                
        yf = torch.cat((yf1, yf2, yf3, yf4, yf5), 1)
        
        return y2, y3, y4, y5, yf    
    


    def init_weights(self):
        #load VGG weights
        for i, (param, pretrained) in enumerate(zip(self.parameters(), 
                                                torchvision.models.vgg16(pretrained=True).parameters())):
            if i < 26:
                param.data = pretrained.data
        
        #initialize other parameters        
        nn.init.normal_(self.dsn2[0].weight, 0, 0.01)
        nn.init.normal_(self.dsn3[0].weight, 0, 0.01)
        nn.init.normal_(self.dsn4[0].weight, 0, 0.01)
        nn.init.normal_(self.dsn5[0].weight, 0, 0.01)
                                
        nn.init.constant_(self.fusion1.weight, 0.25)
        nn.init.constant_(self.fusion2.weight, 0.25)
        nn.init.constant_(self.fusion3.weight, 0.33)
        nn.init.constant_(self.fusion4.weight, 0.5)
        nn.init.constant_(self.fusion5.weight, 1)
        
class SkeletonTrainingSet(Dataset):
    def __init__(self, lst_file, root_dir='', resize=None, transform=None, threshold=None):
        with open(lst_file) as f:
            self.lst = f.read().splitlines()
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize
        self.threshold = threshold
        
    def __len__(self):
        return len(self.lst)
    
    def __getitem__(self, index):
        filenames = self.lst[index].split(" ")
        image = Image.open(self.root_dir + filenames[0])
        label = Image.open(self.root_dir + filenames[1])
        
        if self.resize:
            image = self.resize(image)
            label = self.resize(label)
            
        if self.transform:
            image = self.transform(image)

        #only use single channel for label
        label = np.array(label)   
        if label.ndim == 3:
            label = label[:,:,0]
            
        if self.threshold:
            label_bin_lst = []
            for i in range(len(self.threshold) - 1):
                    label_bin = ((1.2 * label > self.threshold[i]) & (1.2 * label < self.threshold[i+1])).astype(int)
                    label_bin_lst.append(label_bin * (i + 1))
                    
            label2 = label_bin_lst[0]
            label3 = label2 + label_bin_lst[1]
            label4 = label3 + label_bin_lst[2]
            label5 = label4 + label_bin_lst[3]

        return image, label2, label3, label4, label5


class SkeletonTestSet(Dataset):
    def __init__(self, im_dir, root_dir='', resize=None, transform=None):
        self.lst = os.listdir(root_dir + im_dir)
        self.root_dir = root_dir
        self.im_dir = im_dir
        self.resize = resize
        self.transform = transform
        
    def __len__(self):
        return len(self.lst)
    
    def __getitem__(self, index):
        image = Image.open(self.root_dir + self.im_dir + self.lst[index])
            
        if self.resize:
            image = self.resize(image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, self.lst[index]
        

def main():
    global args

    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    model = Skeleton()
    
    if args.mode == 'gpu':           
        torch.cuda.set_device(args.gpuid) 
        torch.cuda.manual_seed(0)               
        model.cuda()
    
    train_dataset = SkeletonTrainingSet(lst_file="aug_data/train_pair.lst", 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])]),
                                        threshold=[0, 14, 40, 92, 196])
    
    test_dataset = SkeletonTestSet(im_dir='images/test/', 
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])]))
        
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.Adam([{'params': model.conv1.parameters()},
                                  {'params': model.conv2.parameters()},
                                  {'params': model.conv3.parameters()},
                                  {'params': model.conv4.parameters()},
                                  {'params': model.conv5.parameters()}, 
                                  {'params': model.dsn2.parameters()},
                                  {'params': model.dsn3.parameters()},
                                  {'params': model.dsn4.parameters()},
                                  {'params': model.dsn5.parameters()}, 
                                  {'params': model.fusion1.parameters()}, 
                                  {'params': model.fusion2.parameters()},
                                  {'params': model.fusion3.parameters()},
                                  {'params': model.fusion4.parameters()},
                                  {'params': model.fusion5.parameters()}], 
                                 lr=args.lr, weight_decay=args.decay) 
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    if args.train:        
        train(model, train_loader, optimizer, scheduler)        
        torch.save(model.state_dict(), 'skeleton.pt')
    
    if args.visualize:    
        visualize(model, test_dataset)
    
    if args.test:
        test(model, test_dataset)

    
def loss_class(y, label):
    label_ = label.cpu().data.numpy()
    count_lst = []       
    for i in range(y.shape[1]):
        n = (label_ == i).sum()
        if n != 0:
            count_lst.append(1/n)
        else:
            count_lst.append(0)
    s = sum(count_lst)
    for i in range(len(count_lst)):
        count_lst[i] = count_lst[i]/s
        
    if args.mode == 'gpu':
        loss = nn.CrossEntropyLoss(torch.cuda.FloatTensor(count_lst))
    else:
        loss = nn.CrossEntropyLoss(torch.FloatTensor(count_lst))
    return loss(y, label)
    
   
def train(model, train_loader, optimizer, scheduler): 
    if args.mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype_float = torch.FloatTensor
        dtype_long = torch.LongTensor
            
    for epoch in range(args.epochs):
        
        optimizer.zero_grad()
        loss_value = 0
    
        for i, (image, label2, label3, label4, label5) in enumerate(train_loader):
            
            image = Variable(image, requires_grad=False).type(dtype_float)
            label2 = Variable(label2, requires_grad=False).type(dtype_long)
            label3 = Variable(label3, requires_grad=False).type(dtype_long)
            label4 = Variable(label4, requires_grad=False).type(dtype_long)
            label5 = Variable(label5, requires_grad=False).type(dtype_long)
            
            y2, y3, y4, y5, yf = model(image)
            
            loss = (loss_class(yf, label5) + loss_class(y2, label2) + loss_class(y3, label3) + \
                    loss_class(y4, label4) + loss_class(y5, label5))/args.itersize
            
            loss_value += loss.cpu().data.numpy()
                    
            loss.backward()
            
            if (i+1) % (args.printfreq * args.itersize) == 0:
                print("epoch: %d    iteration: %d    loss: %.3f" 
                      %(epoch, i//args.itersize, loss_value))
            
            if (i+1) % args.itersize == 0:
                optimizer.step()
                optimizer.zero_grad()
                loss_value = 0
                
        #scheduler.step()

def visualize(model, visualize_dataset):
    if args.mode == 'cpu':
        model.load_state_dict(torch.load('skeleton.pt', 
                                     map_location={'cuda:0':'cpu', 'cuda:1':'cpu',                                                    
                                                   'cuda:2':'cpu', 'cuda:3':'cpu'}))
    else:
        model.load_state_dict(torch.load('skeleton.pt'))
        
    if args.mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
        
    image = visualize_dataset[198][0]
    image = image.unsqueeze(0) 
    image_var = Variable(image, requires_grad=False).type(dtype_float) 
    y2, y3, y4, y5, yf = model(image_var)
    
    yf_ = 1 - F.softmax(yf[0], 0)[0].cpu().data.numpy()
    scale_lst = [yf_]
    plot_single_scale(scale_lst, 22)

def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(1,5,i+1)
        plt.imshow(scale_lst[i], cmap = plt.cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()


def test(model, test_dataset):
    if args.mode == 'cpu':
        model.load_state_dict(torch.load('skeleton.pt', 
                                     map_location={'cuda:0':'cpu', 'cuda:1':'cpu',                                                    
                                                   'cuda:2':'cpu', 'cuda:3':'cpu'}))
    else:
        model.load_state_dict(torch.load('skeleton.pt'))
        
    if args.mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
        
    for i in range(len(test_dataset)):
        image, name = test_dataset[i]
        image = image.unsqueeze(0) 
        if image.shape[1] == 1:
            image = torch.cat((image, image, image), 1)
        
        image_var = Variable(image, requires_grad=False).type(dtype_float)   
        y2, y3, y4, y5, yf = model(image_var)
    
        yf_ = 1 - F.softmax(yf[0], 0)[0].cpu().data.numpy()
        yf_ = yf_/yf_.max()
        
        scipy.misc.imsave('results/' + name[0:-4] + '.png', yf_)
        print('%d of %d images saved' %(i+1, len(test_dataset)))
            
       
if __name__ == '__main__':
    main()                

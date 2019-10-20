import numpy as np
import torch 
import torch.nn as nn
import os
import sys
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import copy
import pickle


from optparse import OptionParser

from util import make_one_hot
from dataset import SampleDataset
from model import UNet, SegNet, DenseNet
from loss import dice_score


def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/samples', help='data path')
    parser.add_option('--model_path', dest='model_path',type='string',
                      default='checkpoints/', help='model_path')
    parser.add_option('--classes', dest='classes', default=28, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=1, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=256, type='int',
                      help='image height')
    parser.add_option('--model', dest='model', type='string',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--gpu', dest='gpu',type='string',
                      default='gpu', help='gpu or cpu')
    parser.add_option('--device1', dest='device1', default=0, type='int',
                      help='device1 index number')
    parser.add_option('--device2', dest='device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--device3', dest='device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--device4', dest='device4', default=-1, type='int',
                      help='device4 index number')

    (options, args) = parser.parse_args()
    return options


def test(model, args):
    
    data_path = args.data_path
    gpu = args.gpu
    n_classes = args.classes
    data_width = args.width
    data_height = args.height
    
    # set device configuration
    device_ids = []
    
    if gpu == 'gpu' :
        
        if not torch.cuda.is_available() :
            print("No cuda available")
            raise SystemExit
            
        device = torch.device(args.device1)
        
        device_ids.append(args.device1)
        
        if args.device2 != -1 :
            device_ids.append(args.device2)
            
        if args.device3 != -1 :
            device_ids.append(args.device3)
        
        if args.device4 != -1 :
            device_ids.append(args.device4)
        
    
    else :
        device = torch.device("cpu")
    
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids = device_ids)
        
    model = model.to(device)
    
    # set testdataset
        
    test_dataset = SampleDataset(data_path)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
    
    avg_score = 0.0
    
    # test
    
    model.eval()   # Set model to evaluate mode
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
        
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            
            target = make_one_hot(labels[:,0,:,:], n_classes, device)
            
            pred = model(inputs)
            
            loss = dice_score(pred,target)
            
            avg_score += loss.data.cpu().numpy()
            
            del inputs, labels, target, pred, loss
            
    avg_score /= len(test_loader)
    
    print('dice_score : {:.4f}'.format(avg_score))
    
if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    model = None
    
    if args.model == 'UNet':
        model = UNet(in_channels = n_channels, n_classes = n_classes)
    
    elif args.model == 'SegNet':
        model = SegNet(in_channels = n_channels, n_classes = n_classes)
        
    elif args.model == 'DenseNet':
        model = DenseNet(in_channels = n_channels, n_classes = n_classes)
    
    else :
        print("wrong model : must be UNet, SegNet, or DenseNet")
        raise SystemExit
        
    summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
    
    model.load_state_dict(torch.load(args.model_path))
    
    test(model, args)
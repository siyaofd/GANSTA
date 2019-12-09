from torchvision.utils import save_image

from PIL import Image
from torchvision.transforms import ToTensor
from facenet_pytorch import MTCNN, InceptionResnetV1

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Input directory
input_dir = '/home/ubuntu/Data/AllAge_Upload/11-15/femaleDir' 
# GPU vs CPU
#compute = 'cuda'
compute = torch.device("cuda:0")
batch_size = 32
num_workers = 4
outputFile = 'tensorAsian11Female.pt'
outputFileFinal = 'tensorAsianPreteenCombined.pt'


#mtcnn = MTCNN(image_size=128) # Must be a large number, else resnet can't take

#ToPIL = transforms.ToPILImage()

#CROP = mtcnn
# Referenced online examples for loader

def main():
    resnet =  InceptionResnetV1(pretrained='vggface2').eval().to(compute)
    dataset = dset.ImageFolder(root=input_dir,
                           transform=transforms.Compose([
                               #mtcnn,     # A callable to crop face
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor()
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(num_workers))
    output_tensors = list()
    print("encoding  data")
    for i, data in enumerate(dataloader, 0):
        image = data[0].to(compute) # use cpu for now
        embedding = resnet(image)
        print(i, embedding.shape)
        output_tensors.append(embedding)

    final_tensor = torch.cat(output_tensors, 0)
    torch.save(final_tensor, outputFile)

def concat():
  t1 = torch.load('tensorAsianPreteenFemale.pt')
  t2 = torch.load('tensorAsianPreteenMale.pt') 
  final_tensor = torch.cat((t1, t2), 0)
  final_tensor = final_tensor[torch.randperm(final_tensor.size()[0])]
  torch.save(final_tensor, outputFileFinal)

if __name__== "__main__":
  main()
  #concat()
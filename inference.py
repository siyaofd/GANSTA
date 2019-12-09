from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data
import torchvision.utils
import torchvision.datasets
import torchvision.transforms as transforms
from torchvision import models
import tensorflow as tf
from facenet_pytorch import InceptionResnetV1
from gan_gpu_wgan import Generator

inferenceInputPath = '/data/Training_Data/UTK_Upload/6-10/femaleDir' #add inference path
inferenceOutputPath = '/data/Output_Data/inference_output/utkFemaleWgan16' #add inference path
aggregatedInferenceOutput = '/data/Output_Data/inference_output/aggregatedWgan1620'
modelCheckpointPathG = 'generatorWgan1620.pt'
device = torch.device("cuda:0")
imageSize = 128
batchSize = 6
randomDimension = 128

def inference():
    seed = 78
    random.seed(seed)
    torch.manual_seed(seed)

    encoding_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    generator = torch.load(modelCheckpointPathG).to(device)
    dataloader = prepareDataLoader(inferenceInputPath)
    for i, data in enumerate(dataloader, 0):
        childhoodImages = data[0].to(device)
        child_encoding = encoding_extractor(childhoodImages).unsqueeze_(-1).unsqueeze_(-2).to(device)
        randomness = torch.randn(batchSize, randomDimension, 1, 1).to(device)
        childhoodImagesFinal = torch.cat((child_encoding, randomness), 1).to(device)
        generatedImages = generator(childhoodImagesFinal)
        res = torch.cat((childhoodImages, generatedImages), 2)
        #print("doing inference")
        torchvision.utils.save_image(generatedImages, '%s/generated_%03d.png' % (inferenceOutputPath, i), normalize=True)  
        torchvision.utils.save_image(res, '%s/generated_%03d.png' % (aggregatedInferenceOutput, i), normalize=True) 

'''
Preprocess images and load them in batchSize
'''
def prepareDataLoader(dataPath):
    dataset = torchvision.datasets.ImageFolder(root=dataPath,
        transform = transforms.Compose(
            [transforms.Resize(imageSize),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 0)
    return dataloader

if __name__== "__main__":
  inference()
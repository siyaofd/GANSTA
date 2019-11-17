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

#high-level configs and hyperparameters
batchSize = 64
dataPath = ### Directory for training image for Discriminator
outputPath = ### Output dir
maxIteration = 50
imageSize = 64
learningRate = 0.0002
l2LossLambd = 0.25

'''
NOTE: we used the idea from this documentation in building the initial architecture 
of this Generator/Discriminator networks: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

We plan to continue tuning the network architectures in the next iterations
'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(192, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
'''
First 6 layers of AlexNet
https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
Use pretrained weights. Added 1 more avgpooling layer at the end, to summarize
features maps for each channel. output 192x1x1 tensor
'''       
class AlexNetConv4(nn.Module):
    def __init__(self):
        super(AlexNetConv4, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            *list(original_model.features[0:6])
        )
        # average pooling for each feature map
        self.averagelayer = torch.nn.AvgPool2d(3)

    def forward(self, x):
        x = self.features(x)
        x = self.averagelayer.forward(x)
        return x

def main():
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    dataloader = prepareDataLoader()
    realLabel = 1
    fakeLabel = 0

    #create generator and discriminator networks
    generator = Generator()
    generator.apply(initializeWeights)
    discriminator = Discriminator()
    discriminator.apply(initializeWeights)
    alexnet_extractor = AlexNetConv4() # extract lower level features as image encoding
    children_encoding = loadTensors('tensor.pt') # load pre-generated tensor
    children_enc_idx = 0

    #define loss function: crossEntropyLoss and optimizer: adam
    criterion = nn.BCELoss()
    optimizerD = optimizer.Adam(discriminator.parameters(), lr=learningRate, betas=(0.5, 0.999))
    optimizerG = optimizer.Adam(generator.parameters(), lr=learningRate, betas=(0.5, 0.999))

    for epoch in range(maxIteration):
        for i, data in enumerate(dataloader, 0):
            realImages = data[0]
            trainingBatchSize = realImages.size(0)
            # Load encodings of child images
            childhoodImages, children_enc_idx = loadChildhoodImages(children_encoding, trainingBatchSize, children_enc_idx)

            #train discriminator
            discriminator.zero_grad()
            label = torch.full((trainingBatchSize,), realLabel)
            output = discriminator(realImages)
            '''
            Our customzied loss function to compute 
            -(ylog(1-D(G(z))) + ylog(D(x)))
            '''
            errorDiscriminatorReal = criterion(output, label)
            errorDiscriminatorReal.backward()

            fake = generator(childhoodImages)
            label.fill_(fakeLabel)
            output = discriminator(fake.detach())
            errorDiscriminatorFake = criterion(output, label)
            errorDiscriminatorFake.backward()
            optimizerD.step()

            #train generator
            generator.zero_grad()
            label.fill_(realLabel)
            output = discriminator(fake)

            output_encoding = alexnet_extractor.forward(fake)
            '''
            Our customzied loss function to compute 
            log(1-D(G(z))) + ||Encode(z) - Encode(child)||2
            '''
            errorGenerator = criterion(output, label) + computeL2Loss(output_encoding, childhoodImages, l2LossLambd)
            errorGenerator.backward()
            optimizerG.step()

            print('Discriminator loss: %.5f Generator loss: %.5f' % (
                (errorDiscriminatorReal + errorDiscriminatorFake).item(), errorGenerator.item()))
            if i % 100 == 0:
                torchvision.utils.save_image(realImages, '%s/real.png' % outputPath, normalize=True)
                torchvision.utils.save_image(generator(childhoodImages).detach(), '%s/generated_epoch_%03d.png' % (outputPath, epoch), normalize=True)
        print("End of Epoch [%d] training" % (epoch))

'''
Preprocess images and load them in batchSize
'''
def prepareDataLoader():
    dataset = torchvision.datasets.ImageFolder(root=dataPath,
        transform = transforms.Compose(
            [transforms.Resize(imageSize),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)
    return dataloader

'''
||Encode(z) - Encode(child)||2
L2 loss between generated pictures and childhood baselines
TODO: change placeholder to use real L2 Loss
'''
def computeL2Loss(output_encoding, childhoodImages, lambd):
    loss = nn.MSELoss()
    return lambd * loss(output_encoding, childhoodImages)

'''
Load childhood images, currently placeholder
In the future they will be dimension 192 vectors output from encoder networks
returns image tensor and next idx
'''
def loadChildhoodImages(encodingTensor, trainingBatchSize, idx):
    len_encoding = encodingTensor.shape[0]
    assert len_encoding > trainingBatchSize      # input tensor size > 1 batch size
    end_idx = (idx + trainingBatchSize) % len_encoding
    # if we wrapped around
    if end_idx < idx:
        return torch.cat((encodingTensor[idx : len_encoding], encodingTensor[0 : end_idx]), 0), end_idx
    else:
        #did not wrap around
        return encodingTensor[idx : end_idx], end_idx

'''
Load precalcualted tensor from file
'''
def loadTensors(filename):
    return torch.load(filename)    

def initializeWeights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

if __name__== "__main__":
  main()
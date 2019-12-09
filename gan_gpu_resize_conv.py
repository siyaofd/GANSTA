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

#high-level configs and hyperparameters
batchSize = 32
trainingDataPath = '/data/Training_Data/utk_small/'### Directory for training image for Discriminator
outputPath = '/data/Output_Data/utk' ### Output dir
inferenceInputPath = None #add inference path
inferenceOutputPath = None #add inference path
modelCheckpointPathG = '/data/Output_Data/model/generatorResize.pt'
modelCheckpointPathD = '/data/Output_Data/model/discriminatorResize.pt'
maxIteration = 500
imageSize = 128
learningRateG = 0.002
learningRateD = 0.0002
l2LossLambd = 0.25
randomDimension = 128
gpu = True
device = None
if gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
ngpu = 1

# help with upsample. Reference: https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588/2
class Interpolate(nn.Module):
    def __init__(self, out_dims, mode = 'nearest'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = out_dims
        self.mode = mode
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Assume input is (160, 2, 2)
        self.main = nn.Sequential(
            Interpolate(out_dims=(4, 4)),
            same_conv2d(in_dims=(160, 4, 4), out_dims=(128, 4, 4), kernel=5),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            Interpolate(out_dims=(8, 8)),
            same_conv2d(in_dims=(128, 8, 8), out_dims=(64, 8, 8), kernel=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            Interpolate(out_dims=(16, 16)),
            same_conv2d(in_dims=(64, 16, 16), out_dims=(32, 16, 16), kernel=5),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            Interpolate(out_dims=(32, 32)),
            same_conv2d(in_dims=(32, 32, 32), out_dims=(16, 32, 32), kernel=5),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            Interpolate(out_dims=(64, 64)),
            same_conv2d(in_dims=(16, 64, 64), out_dims=(8, 64, 64), kernel=5),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            Interpolate(out_dims=(128, 128)),
            same_conv2d(in_dims=(8, 128, 128), out_dims=(3, 128, 128), kernel=5),
            nn.Tanh()
        )

    def forward(self, input):
        res = input.view(input.size(0), 160, 2, 2)
        if gpu:
            output = nn.parallel.data_parallel(self.main, res, range(ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
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
        if gpu:
            output = nn.parallel.data_parallel(self.main, input, range(ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def train(loadCheckpoint=False):
    seed = 78
    random.seed(seed)
    torch.manual_seed(seed)
    dataloader = prepareDataLoader(trainingDataPath)
    realLabel = 1
    fakeLabel = 0
    writer = tf.summary.FileWriter("tf_log")

    #create generator and discriminator networks
    if loadCheckpoint:
        generator = torch.load(modelCheckpointPathG).to(device)
        discriminator = torch.load(modelCheckpointPathD).to(device)
    else:
        generator = Generator().to(device)
        generator.apply(initializeWeights)
        discriminator = Discriminator().to(device)
        discriminator.apply(initializeWeights)
    encoding_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    children_encoding = loadTensors('tensor.pt').to(device) # load pre-generated tensor
    children_enc_idx = 0

    #define loss function: crossEntropyLoss and optimizer: adam
    criterion = nn.BCELoss()
    optimizerD = optimizer.Adam(discriminator.parameters(), lr=learningRateD, betas=(0.5, 0.999))
    optimizerG = optimizer.Adam(generator.parameters(), lr=learningRateG, betas=(0.5, 0.999))

    for epoch in range(maxIteration):
        for i, data in enumerate(dataloader, 0):
            realImages = data[0].to(device)
            trainingBatchSize = realImages.size(0)
            # Load encodings of child images
            childhoodImages, children_enc_idx = loadChildhoodImages(children_encoding, trainingBatchSize, children_enc_idx)
            childhoodImages = childhoodImages.unsqueeze_(-1).unsqueeze_(-2).to(device)
            randomness = torch.randn(trainingBatchSize, randomDimension, 1, 1).to(device)
            childhoodImagesFinal = torch.cat((childhoodImages, randomness), 1).to(device)

            #train discriminator
            discriminator.zero_grad()
            label = torch.full((trainingBatchSize,), realLabel, device=device)
            output = discriminator(realImages)

            '''
            Our customzied loss function to compute 
            -(ylog(1-D(G(z))) + ylog(D(x)))
            '''
            errorDiscriminatorReal = criterion(output, label)
            errorDiscriminatorReal.backward()

            fake = generator(childhoodImagesFinal)
            label.fill_(fakeLabel)
            output = discriminator(fake.detach())
            errorDiscriminatorFake = criterion(output, label)
            errorDiscriminatorFake.backward()
            optimizerD.step()

            #train generator
            generator.zero_grad()
            label.fill_(realLabel)
            output = discriminator(fake)

            output_encoding = encoding_extractor(fake)
            '''
            Our customzied loss function to compute 
            log(1-D(G(z))) + ||Encode(z) - Encode(child)||2
            '''
            errorGenerator = criterion(output, label) + computeL2Loss(output_encoding, childhoodImages, l2LossLambd)
            errorGenerator.backward()
            optimizerG.step()

            print('Discriminator loss: %.5f Generator loss: %.5f' % (
                (errorDiscriminatorReal + errorDiscriminatorFake).item(), errorGenerator.item()))
            summary = tf.Summary(value=[tf.Summary.Value(tag="errorG", simple_value=errorGenerator.item())])
            writer.add_summary(summary, i)
            writer.flush()
            if i % 100 == 0:
                torchvision.utils.save_image(realImages, '%s/real.png' % outputPath, normalize=True)
                torchvision.utils.save_image(generator(childhoodImagesFinal).detach(), '%s/generated_epoch_%03d.png' % (outputPath, epoch), normalize=True)
        print("End of Epoch [%d] training" % (epoch))
        #save current epoch model checkpoint
        torch.save(generator, modelCheckpointPathG)
        torch.save(discriminator, modelCheckpointPathD)

def inference():
    generator = torch.load(modelCheckpointPathG).to(device)
    dataloader = prepareDataLoader(inferenceInputPath)
    for i, data in enumerate(dataloader, 0):
        childhoodImages = data[0].to(device)
        generatedImages = generator(childhoodImages)
        torchvision.utils.save_image(generatedImages, '%s/real.png' % inferenceOutputPath, normalize=True)

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

# Because we want same convolution, should only use stride = 1 shit
def same_conv2d(in_dims, out_dims, kernel, groups=1, dilation = 1, bias=True):
    c_in, h_in, w_in = in_dims
    c_out, h_out, w_out = out_dims
    if h_in != w_in or h_out != w_out:
        raise Exception('H, W dimension not equal: h_in: {}, w_in: {}, h_out: {}, w_out: {}'.format(h_in, w_in, h_out, w_out))

    if kernel % 2 is 0:
        raise Exception('Kernel size must be odd, but is actually ' + str(kernel))
    pad = (kernel + (kernel - 1) * dilation) // 2
    return torch.nn.Conv2d(c_in, c_out, kernel_size = kernel, dilation = dilation, padding = pad, groups = groups, bias = bias)

if __name__== "__main__":
  train()

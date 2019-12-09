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
from facenet_pytorch import InceptionResnetV1
from torch.utils.tensorboard import SummaryWriter
import shutil

#high-level configs and hyperparameters
#tensorboard --logdir='./logs' --port=6006
batchSize = 32
trainingDataPath = '/data/Training_Data/UTK_Upload/26-30/combined'
outputPath = '/data/Output_Data/utk_vanilla_mostrecent' ### Output dir
inferenceInputPath = None #add inference path
inferenceOutputPath = None #add inference path
modelCheckpointPathG = '/data/Output_Data/model/generatorVanilla.pt'
modelCheckpointPathD = '/data/Output_Data/model/discriminatorVanilla.pt'
maxIteration = 500
imageSize = 128
learningRateGInit = 0.0002
learningRateDInit = 0.0001
l2LossLambd = 0.005
randomDimension = 128
decay_rate = 0.25
gpu = True
device = None
if gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
ngpu = 1
enocodingPretrained = "tensorUtk610Combined.pt"
logDir = "tf_log_vanilla"

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

        self.main = nn.Sequential(
            nn.ConvTranspose2d(640, 64 * 8, 4, 1, 0, bias=False),
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
            #nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            AddConv(in_dims=(64, 32, 32), out_dims=(32, 64, 64), kernel=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            AddConv(in_dims=(32, 64, 64), out_dims=(3, 128, 128), kernel=4, stride=2),
            nn.Tanh()
        )

    def forward(self, input):
        if gpu:
            output = nn.parallel.data_parallel(self.main, input, range(ngpu))
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
    shutil.rmtree(logDir) 
    writer = SummaryWriter(logDir)

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
    children_encoding = loadTensors(enocodingPretrained).to(device) # load pre-generated tensor
    children_enc_idx = 0

    #define loss function: crossEntropyLoss and optimizer: adam
    criterion = nn.BCELoss()
    optimizerD = optimizer.Adam(discriminator.parameters(), lr=learningRateDInit, betas=(0.5, 0.999))
    optimizerG = optimizer.Adam(generator.parameters(), lr=learningRateGInit, betas=(0.5, 0.999))

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
            encoding_loss = computeL2Loss(output_encoding, childhoodImages, l2LossLambd)
            errorGenerator = criterion(output, label) + encoding_loss
            errorGenerator.backward()
            optimizerG.step()

            print('Discriminator loss: %.5f Generator loss: %.5f' % (
                (errorDiscriminatorReal + errorDiscriminatorFake).item(), errorGenerator.item(), encoding_loss.item()))
            if i % 100 == 0:
                torchvision.utils.save_image(realImages, '%s/real.png' % outputPath, normalize=True)
                torchvision.utils.save_image(generator(childhoodImagesFinal).detach(), '%s/generated_epoch_%03d.png' % (outputPath, epoch), normalize=True)
                writer.add_scalars('age_gan', {'LossD':(errorDiscriminatorFake + errorDiscriminatorReal).item(),
                    'LossG': errorGenerator.item()}, i)
        print("End of Epoch [%d] training" % (epoch))
        #learning rate decay
        if epoch % 30 == 0  and epoch != 0:
            #step  decay
            learningRateD = learningRateDInit * (0.25 ** (epoch/30))
            learningRateG = learningRateGInit * (0.25 ** (epoch/30))
            optimizerD = optimizer.Adam(discriminator.parameters(), lr=learningRateD, betas=(0.5, 0.999))
            optimizerG = optimizer.Adam(generator.parameters(), lr=learningRateG, betas=(0.5, 0.999))

        #save current epoch model checkpoint
        torch.save(generator, modelCheckpointPathG)
        torch.save(discriminator, modelCheckpointPathD)

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

'''
||Encode(z) - Encode(child)||2
L2 loss between generated pictures and childhood baselines
'''
def computeL2Loss(output_encoding, childhoodImages, lambd):
    loss = nn.MSELoss()
    return lambd * loss(output_encoding, childhoodImages)

def loadChildhoodImages(encodingTensor, trainingBatchSize, idx):
    len_encoding = encodingTensor.shape[0]
    assert len_encoding > trainingBatchSize      # input tensor size > 1 batch size
    end_idx = (idx + trainingBatchSize) % len_encoding
    # if we wrapped around
    if end_idx < idx:
        res = torch.cat((encodingTensor[idx : len_encoding], encodingTensor[0 : end_idx]), 0), 0
        shuffleTensorEncoding(encodingTensor)
        return res
    else:
        #did not wrap around
        return encodingTensor[idx : end_idx], end_idx

'''
Shuffle encoding to prevent overfitting
'''
def shuffleTensorEncoding(encodingTensor):
    encodingTensor=encodingTensor[torch.randperm(encodingTensor.size()[0])]

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

def AddConv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)

    c_in, h_in, w_in = in_dims
    c_out, h_out, w_out = out_dims

    padding = [0, 0]
    output_padding = [0, 0]

    lhs_0 = -h_out + (h_in - 1) * stride[0] + kernel[0]  # = 2p[0] - o[0]
    if lhs_0 % 2 == 0:
        padding[0] = lhs_0 // 2
    else:
        padding[0] = lhs_0 // 2 + 1
        output_padding[0] = 1

    lhs_1 = -w_out + (w_in - 1) * stride[1] + kernel[1]  # = 2p[1] - o[1]
    if lhs_1 % 2 == 0:
        padding[1] = lhs_1 // 2
    else:
        padding[1] = lhs_1 // 2 + 1
        output_padding[1] = 1

    print("padding: " )
    print(padding)
    print("output padding: ")
    print(output_padding)
    return torch.nn.ConvTranspose2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel,
        stride=stride,
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        groups=groups,
        bias=bias,
        dilation=dilation
    )
    
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

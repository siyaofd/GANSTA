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
from torch.utils.tensorboard import SummaryWriter
from facenet_pytorch import InceptionResnetV1
import shutil
from torch.autograd import Variable
from torch import autograd

#high-level configs and hyperparameters
batchSize = 32
trainingDataPath = '/data/Training_Data/UTK_Upload/26-30/combined'### Directory for training image for Discriminator
outputPath = '/data/Output_Data/utk_wgangp' ### Output dir
inferenceInputPath = None #add inference path
inferenceOutputPath = None #add inference path
modelCheckpointPathG = '/data/Output_Data/model/generatorWganGP.pt'
modelCheckpointPathD = '/data/Output_Data/model/discriminatorWganGP.pt'
maxIteration = 500000
discriminatorIter = 1
imageSize = 128
learningRateG = 0.00005
learningRateD = 0.00005
l2LossLambd = 0.25
randomDimension = 128
weight_cliping_limit = 0.02
gpu = True
lambdaT = 10
device = None
if gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
ngpu = 1
logDir = "tf_log_wagn_gp"
enocodingPretrained = "tensorUtk610Combined.pt"

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

'''
NOTE: we used the idea from this documentation in building the initial architecture 
of this Generator/Discriminator networks: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

We plan to continue tuning the network architectures in the next iterations

Resize convolution instead of deconvolution to avoid checkerboard artifact
We first upsample and then do regular convolution. The operations are equivalent.

'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 64 * 8, 4, 1, 0, bias=False),
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
            addDeconv(in_dims=(64, 32, 32), out_dims=(32, 64, 64), kernel=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            addDeconv(in_dims=(32, 64, 64), out_dims=(3, 128, 128), kernel=4, stride=2),
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
        )

    def forward(self, input):
        if gpu:
            output = nn.parallel.data_parallel(self.main, input, range(ngpu))
        else:
            output = self.main(input)
        return output

def train(loadCheckpoint=False):
    seed = 78
    random.seed(seed)
    torch.manual_seed(seed)
    dataloader = getDataFromLoader()
    shutil.rmtree(logDir) 
    writer = SummaryWriter(logDir)

    #create generator and discriminator networks
    if loadCheckpoint:
        generator = torch.load(modelCheckpointPathG).to(device)
        discriminator = torch.load(modelCheckpointPathD).to(device)
    else:
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
    encoding_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    children_encoding = loadTensors(enocodingPretrained).to(device) # load pre-generated tensor
    children_enc_idx = 0

    optimizerD = optimizer.RMSprop(discriminator.parameters(), lr=learningRateD)
    optimizerG = optimizer.RMSprop(generator.parameters(), lr=learningRateG)

    for g_iter in range(maxIteration):
        for d_iter in range(discriminatorIter):   
            realImages = dataloader.__next__()
            trainingBatchSize = realImages.size(0)
            # Load encodings of child images
            childhoodImages, children_enc_idx = loadChildhoodImages(children_encoding, trainingBatchSize, children_enc_idx)
            childhoodImages = childhoodImages.unsqueeze_(-1).unsqueeze_(-2).to(device)
            #randomness = torch.randn(trainingBatchSize, randomDimension, 1, 1).to(device)
            #childhoodImagesFinal = torch.cat((childhoodImages, randomness), 1).to(device)

            discriminator.zero_grad()
            output = discriminator(realImages)
            errorDiscriminatorReal = torch.mean(output)
            fake = generator(childhoodImages)
            output = discriminator(fake.detach())
            errorDiscriminatorFake = torch.mean(output)
            errorD = errorDiscriminatorFake - errorDiscriminatorReal
            errorD.backward()
            optimizerD.step()
            for p in discriminator.parameters():
                p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

        generator.zero_grad()
        output = discriminator(fake)

        #encoding_loss = computeL2Loss(output_encoding, childhoodImages, l2LossLambd)
        errorGenerator = -torch.mean(output)
        errorGenerator.backward()
        optimizerG.step()

        print('Discriminator loss: %.5f Generator loss: %.5f' % (
            errorD.item(), errorGenerator.item()))

        if g_iter % 100 == 0:
            epoch = g_iter / 100
            torchvision.utils.save_image(realImages, '%s/real.png' % outputPath, normalize=True)
            torchvision.utils.save_image(generator(childhoodImages).detach(), '%s/generated_epoch_%03d.png' % (outputPath, epoch), normalize=True)
            writer.add_scalars('wgan', {'LossD':errorD.item(), 'LossG': errorGenerator.item()}, epoch)
            print("End of Epoch [%d] training" % (epoch))
            #save current epoch model checkpoint
            torch.save(generator, modelCheckpointPathG)
            torch.save(discriminator, modelCheckpointPathD)

def getDataFromLoader():
    dataloader = prepareDataLoader(trainingDataPath)
    while True:
        for i, (images, _) in enumerate(dataloader):
            yield images.to(device)

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
        res = torch.cat((encodingTensor[idx : len_encoding], encodingTensor[0 : end_idx]), 0), 0
        shuffleTensorEncoding(encodingTensor)
        return res
    else:
        #did not wrap around
        return encodingTensor[idx : end_idx], end_idx

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

def addDeconv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
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

def calculate_gradient_penalty(batchSize, real_images, fake_images, discriminator):
    eta = torch.FloatTensor(batchSize,1,1,1).uniform_(0,1)
    eta = eta.expand(batchSize, real_images.size(1), real_images.size(2), real_images.size(3)).to(device)
    

    interpolated = eta * real_images + ((1 - eta) * fake_images).to(device)

    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(device),
                               create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambdaT
    return grad_penalty   

if __name__== "__main__":
  train()

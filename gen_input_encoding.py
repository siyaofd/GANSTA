import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

original_model = models.alexnet(pretrained=True) # Use locally cached pretrained weights
dataroot = './young_folder'

# Referenced online examples for loader
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                         shuffle=True, num_workers=int(4))

class AlexNetConv4(nn.Module):
	def __init__(self):
		super(AlexNetConv4, self).__init__()
		self.features = nn.Sequential(
			*list(original_model.features[0:6])
		)
		# average pooling for each feature map
		self.averagelayer = torch.nn.AvgPool2d(3)
	def forward(self, x):
		x = self.features(x)
		x = self.averagelayer.forward(x)
		return x

# First 6 layers of AlexNet for feature extraction
model = AlexNetConv4()

output_tensors = list()
for i, data in enumerate(dataloader, 0):
	real_cpu = data[0].to('cpu') # use cpu for now
	output = model.forward(real_cpu)
	output_tensors.append(output)
final_tensor = torch.cat(output_tensors, 0)
print(final_tensor.shape)

# Save generated children encoding tensors to be reused
torch.save(final_tensor, 'tensor.pt')

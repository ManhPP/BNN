import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import datasets, transforms

from src.model.binary_cnn import BinaryCNN
from src.model.cnn import CNN
from src.model.fc import BinaryFC, FC

model = FC()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoint/" + model._get_name() + ".pth", map_location=device)
state_dict = checkpoint['net']
model.load_state_dict(state_dict, strict=False)

model.train()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

dataiter = iter(train_loader)
inputs, targets = dataiter.next()

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table())

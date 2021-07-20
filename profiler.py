import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import datasets, transforms
from torchsummary import summary

import tracemalloc

from src.model.binary_cnn import BinaryCNN
from src.model.cnn import CNN
from src.model.fc import BinaryFC, FC

# model = BinaryFC()
# model = BinaryCNN()
# model = FC()
model = CNN()

summary(model, (28, 28, 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoint/" + model._get_name() + ".pth", map_location=device)
state_dict = checkpoint['net']
model.load_state_dict(state_dict, strict=False)

model.eval()

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
        tracemalloc.start()
        model(inputs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

print(prof.key_averages().table())
print(f"{current:0.2f}, {peak:0.2f}")

#   14403.00, 29252.00
#   14483.00, 29331.00
#   463114.00, 476873.00
#   462784.00, 476743.00

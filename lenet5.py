import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(42)
batch_size = 100

train_data = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
    self.fc1 = nn.Linear(256, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  
  def forward(self, x):
    out = self.conv1(x)
    out = F.avg_pool2d(F.relu(out), kernel_size=2)
    out = self.conv2(out)
    out = F.avg_pool2d(F.relu(out), kernel_size=2)
    out = out.view(-1, 256)
    out = self.fc1(out)
    out = F.relu(out)
    out = self.fc2(out)
    out = F.relu(out)
    out = self.fc3(out)

    return out

model = LeNet5().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-03)

def Binarize(tensor,is_det=True):
  if is_det:
    return tensor.sign()
  else:
    return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class BinarizeConv2d(nn.Conv2d):
  def __init__(self, *kargs, **kwargs):
    super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
  
  def forward(self, input):
    if input.size(2) != 1:
      input.data = Binarize(input.data)
    if not hasattr(self.weight, 'org'):
      self.weight.org = self.weight.data.clone()
    self.weight.data = Binarize(self.weight.org)

    out = F.conv2d(input, self.weight, None, self.stride,
                   self.padding, self.dilation, self.groups)
    
    return out

class BinaryLeNet5(nn.Module):
  def __init__(self):
    super(BinaryLeNet5, self).__init__()
    self.conv1 = BinarizeConv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
    self.conv2 = BinarizeConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
    self.fc1 = nn.Linear(256, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    out = self.conv1(x)
    out = F.avg_pool2d(out, kernel_size=2)
    out = self.conv2(out)
    out = F.avg_pool2d(out, kernel_size=2)
    out = out.view(-1, 256)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)

    return out

bin_model = BinaryLeNet5().to(device)
bin_optimizer = optim.Adam(bin_model.parameters(), lr=1e-03)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def b2mb(x): return int(x/2**20)
class TorchTracemalloc():
    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used   = b2mb(self.end-self.begin)
        self.peaked = b2mb(self.peak-self.begin)
        print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

epochs = 5
import tracemalloc

tracemalloc.start()
for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train(train_loader, model, loss_fn, optimizer)
  test(test_loader, model, loss_fn)
  torch.save(model.state_dict(), "lenet5.pth")
  print("Done!")
current, peak =  tracemalloc.get_traced_memory()
print(f"{current:0.2f}, {peak:0.2f}")
tracemalloc.stop()

tracemalloc.start()
for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train(train_loader, bin_model, loss_fn, bin_optimizer)
  test(test_loader, bin_model, loss_fn)
  torch.save(bin_model.state_dict(), "binary_lenet5.pth")
  print("Done!")
current, peak =  tracemalloc.get_traced_memory()
print(f"{current:0.2f}, {peak:0.2f}")
tracemalloc.stop()

# with TorchTracemalloc() as tt:
#   for t in range(epochs):
#       print(f"Epoch {t+1}\n-------------------------------")
#       train(train_loader, model, loss_fn, optimizer)
#       test(test_loader, model, loss_fn)
#       torch.save(model.state_dict(), "lenet5.pth")
#   print("Done!")

# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_loader, cnn, loss_fn, cnn_optimizer)
#     test(test_loader, cnn, loss_fn)
# print("Done!")

# with TorchTracemalloc() as tt:
#   for t in range(epochs):
#       print(f"Epoch {t+1}\n-------------------------------")
#       train(train_loader, bin_model, loss_fn, bin_optimizer)
#       test(test_loader, bin_model, loss_fn)
#       torch.save(model.state_dict(), "binary_lenet5.pth")
#   print("Done!")

device = 'cpu'
_model = LeNet5().to(device)
_model.load_state_dict(torch.load("lenet5.pth"))

_bin_model = BinaryLeNet5().to(device)
_bin_model.load_state_dict(torch.load("binary_lenet5.pth"))

(X, y) = next(iter(test_loader))
_model.eval()
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    # with record_function("model_inference"):
    _model(X.to(device))

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

_bin_model.eval()
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    # with record_function("model_inference"):
    _bin_model(X.to(device))

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
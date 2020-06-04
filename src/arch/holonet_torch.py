import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.data.loader import DataLoader
import time
import imageio

class HoloNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=64):
        super(HoloNet, self).__init__()
        p1, p2 = int(kernel_size/2 - 1), int(kernel_size/2)
        self.zero = nn.ZeroPad2d((p1,p2,p1,p2))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv1(self.zero(x))
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kernel_size=64
net = HoloNet(kernel_size=kernel_size)
net.cuda().to(device)

# criterion = nn.Adam()
loss_fn = nn.MSELoss(reduction='sum').cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
batch_size = 16
# load the data
x, y = DataLoader.load_training('ds-lymphoma', separate=False, records=-1)
x, y = torch.from_numpy(x), torch.from_numpy(y[:,:,:,0,:]) # magnitude only
x, y = x.permute(0, 3, 1, 2).float(), y.permute(0, 3, 1, 2).float()  # from NHWC to NCHW
train_dataset = torch.utils.data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
last = time.time()
for epoch in range(150):
    for i, (data, target) in enumerate(train_loader):
        # send data to GPU\n",
        inputs, labels = data.to(device), target.to(device)

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = net(inputs)

        # Compute and print loss.
        loss = loss_fn(y_pred, labels)
        if epoch % 100 == 99:
            print(epoch, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()
        # print('zero_grad')
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()
        # print('backward')
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        # print('step')
        elapsed = time.time() - last
        print('Epoch {0:02d}, Batch {1:02d}/{2:02d}, Time: {3:5.5f}s'.format(epoch, i,
            int(x.shape[0]/batch_size), elapsed))
        last = time.time()
    if epoch % 5 == 0:
        torch.save(net.state_dict(),
                   r'D:\d3-recon-ml\experiments\torch_holo\holonet_{0}_{1:02d}.dict'
                   .format(kernel_size, epoch))
        imageio.imwrite(r'D:\d3-recon-ml\experiments\torch_holo\holonet_{0}_{1:02d}.png'
                   .format(kernel_size, epoch), net.conv1.weight.data.squeeze().cpu().numpy())



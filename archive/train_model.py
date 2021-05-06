import torch
import pickle
import torch.nn as nn
import numpy as np
import torch.optim as optim
from random import sample
import matplotlib.pyplot as plt
import cv2
from augment_image import *


project = "marker"
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(f'{project}.labels', 'rb') as labels_in:
    labels = pickle.load(labels_in)
    #num_keypoints = labels_dict.shape[1]
    num_keypoints = labels.shape[1]
    
    labels.shape
    # Create dictionary temporarily; implement this in annotation script
    labels_dict = {'head': labels[:,0,:], 'body': labels[:,1,:]}
#    labels_dict = {'head': labels[:,0,:], 'body': labels[:,1,:], 'tail': labels[:,2,:]}
    
    print(labels.shape)

with open(f'{project}.rgbd', 'rb') as frames_in:
    frames = pickle.load(frames_in)
    
    # Depth frame normalization and clipping for converting into uint8; implement user input functionality
    clip_dist = 2000
    np.clip(frames[:,:,:,3], 0, clip_dist, frames[:,:,:,3])
    frames[:,:,:,3] = (((frames[:,:,:,3]/clip_dist))*255).astype(np.uint8)
    frames = np.uint8(frames)
    
    #frames = [frames[k] for k in range(len(frames)) if k in labels_dict.keys()]
    
    print(len(frames))

frames = np.array(frames)
labels = np.array(labels)

num_frames = len(labels)
frame_size = frames[0].shape[0]

print(f"Frame Size: {frames.shape}")
print(f"Label Size: {labels.shape}")


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear((30**2) * 32, 512, bias=True),
            nn.LeakyReLU())
        self.drop_out = nn.Dropout()
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.LeakyReLU())
        self.fc3 = nn.Linear(256, num_keypoints * 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        out = self.drop_out(out)
        out = self.fc3(out)
        return out

model_name = 'cnn'
model = DepthNet()
model.to(device)

# Model Hyper-parameters
num_epochs = 500
batch_size = 128
learning_rate = 0.0001
loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

running_loss = float('inf')
loss_history = [float('inf')]
for epoch in range(num_epochs):
    sample_size = min(batch_size, num_frames)
    batch_indices = sample([k for k in range(num_frames)], sample_size)
    frame_batch = np.array([frames[index] for index in batch_indices])
    label_batch = np.array([labels[index] for index in batch_indices])
    print(f'[Epoch: {epoch + 1}/{num_epochs}]\tLoss: {round(running_loss, 3)}')
    running_loss = 0.0

    for i in range(batch_size):
        print("|", end='')
        frame_batch_i = torch.from_numpy(frame_batch).type(torch.float).reshape(-1, 4, frame_size, frame_size)
#        label_batch_i = torch.from_numpy(label_batch).type(torch.float).reshape(-1, 2)
        
        # Reshape to have a format of "num_keypoints * 2" values per image as per predictions
        label_batch_i = torch.from_numpy(label_batch).type(torch.float).reshape(-1, num_keypoints * 2)

        inputs_i, labels_i = frame_batch_i.to(device), label_batch_i.to(device)
        optimizer.zero_grad()

        output = model(inputs_i)
        
#        print(frames.shape, labels.shape, frame_batch.shape, label_batch.shape, frame_batch_i.shape, label_batch_i.shape, output.shape, labels_i.shape)
        
        
        loss = loss_function(output, labels_i)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % batch_size == batch_size - 1:
            print("")
            if running_loss < min(loss_history):
                torch.save(model.state_dict(), f'{project}_{model_name}.net')
            loss_history.append(running_loss)


print('[INFO] Finished Training')
torch.save(model.state_dict(), f'{project}_{model_name}.net')

with open(f'{project}_{model_name}.loss_hist', 'wb') as f:
    np.save(f, loss_history)


plt.loglog(loss_history)
plt.title("Log-Log Loss History (Epoch vs. Loss)")
plt.show()


import torch
import pickle
import torch.nn as nn
import numpy as np
import torch.optim as optim
from random import sample
import matplotlib.pyplot as plt
import cv2
from augment_image import *

from models.ResNet import *
from models.ResNet import resnet


model_save_path = 'training/'
data_path = 'data/'


project = "marker"
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(f'{data_path}{project}.labels', 'rb') as labels_in:
    labels = pickle.load(labels_in)
    #num_keypoints = labels_dict.shape[1]
    num_keypoints = labels.shape[1]
    
    # Create dictionary temporarily; implement this in annotation script
    labels_dict = {'head': labels[:,0,:], 'body': labels[:,1,:]}
#    labels_dict = {'head': labels[:,0,:], 'body': labels[:,1,:], 'tail': labels[:,2,:]}
    
    print(labels.shape)

with open(f'{data_path}{project}.rgbd', 'rb') as frames_in:
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

# Test Validation split
np.random.seed(22)
train_test_split_pct = 0.8
split_idx = int(np.floor(num_frames * train_test_split_pct))

indices = np.random.permutation(num_frames)
training_idx, test_idx = indices[:split_idx], indices[split_idx:]

frames_training, frames_test = frames[training_idx,:], frames[test_idx,:]
labels_training, labels_test = labels[training_idx,:], labels[test_idx,:]


print(f"Training Size:\nFrames: {frames_training.shape} Labels: {labels_training.shape}")
print(f"Test Size:\nFrames: {frames_test.shape} Labels: {labels_test.shape}")

num_frames_train, num_frames_test = len(labels_training), len(labels_test)
frame_size_train, frame_size_test = frames_training[0].shape[0], frames_test[0].shape[0]

model_name = 'resnet_test'
model = resnet(num_classes = num_keypoints * 2)
model.to(device)

# Model Hyper-parameters
num_epochs = 500
batch_size = 32
learning_rate = 0.0001
loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

running_loss = float('inf')
loss_history = [float('inf')]

for epoch in range(num_epochs):
    sample_size = min(batch_size, num_frames_train)
    batch_indices = sample([k for k in range(num_frames_train)], sample_size)
    frame_batch = np.array([frames_training[index] for index in batch_indices])
    label_batch = np.array([labels_training[index] for index in batch_indices])
    print(f'[Epoch: {epoch + 1}/{num_epochs}]\tLoss: {round(running_loss, 3)}')
    running_loss = 0.0

    for i in range(batch_size):
        print("|", end='')
        frame_batch_i = torch.from_numpy(frame_batch).type(torch.float).reshape(-1, 4, frame_size_train, frame_size_train)
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
                torch.save(model.state_dict(), f'{model_save_path}{project}_{model_name}.net')
            loss_history.append(running_loss)


print('[INFO] Finished Training')
torch.save(model.state_dict(), f'{model_save_path}{project}_{model_name}.net')

with open(f'{model_save_path}{project}_{model_name}.loss_hist', 'wb') as f:
    np.save(f, loss_history)


plt.loglog(loss_history)
plt.title("Log-Log Loss History (Epoch vs. Loss)")
plt.show()


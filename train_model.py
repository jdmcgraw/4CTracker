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
from models.DenseNet import *
from models.DenseNet import se_densenet

# Set project folder paths
model_save_path = 'training/'
data_path = 'data/'

# Train test Split (temporal)
np.random.seed(22)
train_test_split_pct = 0.8

# Set project name
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

# Test Validation split index
split_idx = int(np.floor(num_frames * train_test_split_pct))

# Temporal train test split
# Assuming all frames are labelled
labels_training, labels_test = labels[:split_idx,:], labels[split_idx:,:]
frames_training, frames_test = frames[:split_idx,:], frames[split_idx:split_idx + len(labels_test),:]



print(f"Training Size:\nFrames: {frames_training.shape} Labels: {labels_training.shape}")
print(f"Test Size:\nFrames: {frames_test.shape} Labels: {labels_test.shape}")

num_frames_train, num_frames_test = len(labels_training), len(labels_test)
frame_size_train, frame_size_test = frames_training[0].shape[0], frames_test[0].shape[0]

# Select model to train - ResNet (multiple variants), DenseNet (multiple variants)
model_name = 'resnet_train_loss'
model = resnet(num_classes = num_keypoints * 2)
model.to(device)

# Model Hyper-parameters
num_epochs = 500
batch_size = 16
learning_rate = 0.0001
loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

running_loss = float('inf')
loss_history = [float('inf')]
loss_history_val = [float('inf')]


# Validation dataset transformation
frame_test_in = torch.from_numpy(frames_test).type(torch.float).reshape(-1, 4, frame_size_train, frame_size_train)

# Reshape to have a format of "num_keypoints * 2" values per image as per predictions
label_test_in = torch.from_numpy(labels_test).type(torch.float).reshape(-1, num_keypoints * 2)
inputs_test_in, labels_test_in = frame_test_in.to(device), label_test_in.to(device)


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
        # Reshape to have a format of "num_keypoints * 2" values per image as per predictions
        label_batch_i = torch.from_numpy(label_batch).type(torch.float).reshape(-1, num_keypoints * 2)

        inputs_i, labels_i = frame_batch_i.to(device), label_batch_i.to(device)
        optimizer.zero_grad()

        output = model(inputs_i)
        
        loss = loss_function(output, labels_i)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % batch_size == batch_size - 1:
            print("")
            
            # Validation dataset loss
            output_test = model(inputs_test_in)
            loss_test = loss_function(output_test, labels_test_in)            
            if running_loss < min(loss_history):
                torch.save(model.state_dict(), f'{model_save_path}{project}_{model_name}.net')
            loss_history.append(loss)
            loss_history_val.append(loss_test.item())
            print(f'\nTraining Loss: {round(running_loss,5)}\tValidation Loss: {round(loss_test.item(),5)}')
            


print('[INFO] Finished Training')
torch.save(model.state_dict(), f'{model_save_path}{project}_{model_name}.net')

with open(f'{model_save_path}{project}_{model_name}.loss_hist', 'wb') as f:
    np.save(f, loss_history)
    
with open(f'{model_save_path}{project}_{model_name}.val_loss_hist', 'wb') as f:
    np.save(f, loss_history)

plt.loglog(loss_history)
plt.title("Log-Log Train Loss History (Epoch vs. Loss)")
plt.show()

plt.loglog(loss_history_val)
plt.title("Log-Log Validation Loss History (Epoch vs. Loss)")
plt.show()
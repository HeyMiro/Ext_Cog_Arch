#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import torch
from torch import nn
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
import cv2
import datasets
from torchvision import transforms

class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        num_joints = 6

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(37*37*32, num_joints)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

'''
Test of the babbling idea. Training simple network to predict joint state. 
'''
def main():
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Brain().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    babbling_dataset = datasets.build_dataset(transform=transforms.Compose([                                                        
                                                        transforms.ToPILImage(),
                                                        transforms.ToTensor(),
                                                        transforms.Resize((150,150))]))
    train_loader = torch.utils.data.DataLoader(dataset=babbling_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, states) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 2 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))




if __name__ == "__main__":
    main()

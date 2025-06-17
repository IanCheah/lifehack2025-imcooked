import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

data_directory = "data"
original_directory = os.path.join(data_directory, "original")
corrected_directory = os.path.join(data_directory, "corrected")

original_images = sorted(os.listdir(original_directory))
filename = original_images[0]

X = []
y = []
labels = []

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])

for image_name in original_images:
    original_path = os.path.join(original_directory, image_name)
    protanopia_path = os.path.join(corrected_directory, "protanopia", image_name)
    deuteranopia_path = os.path.join(corrected_directory, "deuteranopia", image_name)
    tritanopia_path = os.path.join(corrected_directory, "tritanopia", image_name)
    original_img = transform(Image.open(original_path).convert("RGB"))
    protanopia_img = transform(Image.open(protanopia_path).convert("RGB"))
    deuteranopia_img = transform(Image.open(deuteranopia_path).convert("RGB"))
    tritanopia_img = transform(Image.open(tritanopia_path).convert("RGB"))
    
    for i in range(3):
        X.append(original_img)
        labels.append(i) #0 - protanopia, 1-deuteranopia, 2-tritanopia

    y.append(protanopia_img)
    y.append(deuteranopia_img)
    y.append(tritanopia_img)

X = torch.stack(X)
y = torch.stack(y)
labels = torch.tensor(labels)

X_train, X_test, y_train, y_test, label_train, label_test = train_test_split(X, y, labels, test_size = 0.2)

train_dataset = TensorDataset(X_train, label_train, y_train)
test_dataset = TensorDataset(X_test, label_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class ColorBlindnessCNN(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3 + num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1), 
            nn.Sigmoid()  
        )

    def forward(self, x, labels):
        batch_size, _, h, w = x.size()
        label_onehot = F.one_hot(labels, num_classes=3).float() 
        label_maps = label_onehot.view(batch_size, 3, 1, 1).expand(-1, -1, h, w)
        
        x = torch.cat([x, label_maps], dim=1)  
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = ColorBlindnessCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, targets in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, labels)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels, targets in test_loader:
            inputs, labels, targets = inputs.to(device), labels.to(device), targets.to(device)
            outputs = model(inputs, labels)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(test_loader)}")

    model_path = "cnn_model.pth" + str(epoch)
    torch.save(model.state_dict(), model_path)
    print("Model saved")

    # Testing
    import torchvision.transforms.functional as TF

    output_dir = "model_outputs"
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Take a few samples from the test dataset (e.g., first 5)
        for i in range(min(5, len(test_dataset))):
            input_img, label, target_img = test_dataset[i]
            input_img = input_img.unsqueeze(0).to(device)  # add batch dim
            label = label.unsqueeze(0).to(device)

            output = model(input_img, label)
            output = output.squeeze(0).cpu()  # remove batch dim

            # Convert output tensor (3, H, W) to PIL image
            output_img = TF.to_pil_image(output)

            # Save output image
            output_filename = os.path.join(output_dir, f"output_{i}_label{label.item()}_epoch{epoch+1}.png")
            output_img.save(output_filename)
            print(f"Saved output image to {output_filename}")
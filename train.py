import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cpu')

train_path = 'dentalData/TD-1/Training images'
test_path = 'dentalData/TD-1/Testing images'

# 7 classes for diseases
# Images have no color so input shape should be 1
# Output layer should be 7
# Model will have 2 convolutional layers with 2 fully connected layers
# Using Stochastic Gradient Descent Optimizer and Categorical-Cross-Entropy Loss Function

class_names = ['Pulpitis',  # 0
               'Bony Impaction',  # 1
               'Improper Restoration with Chronic Apical Periodontitis',  # 2
               'Chronic Apical Periodontitis with Vertical Bone Loss',  # 3
               'Embedded Tooth',  # 4
               'Dental Caries',  # 5
               'Periodontitis']  # 6

epochs = 8
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.Resize((500, 500)),  # All images need to be one size, so just an approximate, may change later.
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = next(dataiter)

# imshow(torchvision.utils.make_grid(images))


class DentalModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_output_size = hidden_units * (500 // 4) * (500 // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.conv_output_size,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        print(x.shape)
        x = self.conv_layer2(x)
        print(x.shape)
        x = self.classifier(x)

        return x


model = DentalModel(input_shape=3,
                    hidden_units=7,
                    output_shape=len(class_names)).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# n_total_steps = len(train_loader)
# for epoch in range(epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss_val = loss(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()
#
#         # if (i+1) % 2000 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_val.item():.4f}')

# print("Training Complete")
torch.save(model.state_dict(), 'dental_cnn.pth')


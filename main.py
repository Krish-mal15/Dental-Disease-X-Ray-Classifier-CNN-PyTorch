import torch
from torchvision.transforms import transforms
from PIL import Image
from train import DentalModel  # Import your model class

model = DentalModel(3, 7, 7)
# Parameters of this model
model.load_state_dict(torch.load('dental_cnn.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your new image
new_image_path = 'dentalData/TD-2/Testing images/1.Irreversible pulpitis with Acute periodontitis/1e (1).jpg'
new_image = Image.open(new_image_path)

input_data = transform(new_image).unsqueeze(0)

with torch.no_grad():
    output = model(input_data)

predicted_class = torch.argmax(output, dim=1).item()

print("Predicted class:", predicted_class)


def label(prediction):
    if prediction == 0:
        print("Disease: ", 'Pulpitis')
    if prediction == 1:
        print("Disease: ", 'Bony Impaction')
    if prediction == 2:
        print("Disease: ", 'Improper Restoration with Chronic Apical Periodontitis')
    if prediction == 3:
        print("Disease: ", 'Chronic Apical Periodontitis with Vertical Bone Loss')
    if prediction == 4:
        print("Disease: ", 'Embedded Tooth')
    if prediction == 5:
        print("Disease: ", 'Dental Caries')
    if prediction == 6:
        print("Disease: ", 'Periodontitis')


label(predicted_class)

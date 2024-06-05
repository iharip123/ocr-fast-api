import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

model_path = 'project/model/best_resnet18_model.pth'
num_classes = 56

class CustomResNet18(nn.Module):
  def __init__(self, num_classes=56):
    super(CustomResNet18, self).__init__()
    self.resnet18 = models.resnet18(pretrained=True)
    num_ftrs = self.resnet18.fc.in_features
    self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

  def forward(self, x):
    return self.resnet18(x)

class ModelManager:
  def __init__(self, model_path='project/model/best_resnet18_model.pth', num_classes=56):
    self.model = self._load_model(model_path, num_classes)
    self.device = torch.device('cpu')
    self.model.to(self.device)
    self.model.eval()

  def _load_model(self, model_path='project/model/best_resnet18_model.pth', num_classes=56):
    model = CustomResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

  def predict(self, image_path, img_width=32, img_height=32):
    transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(self.device)

    with torch.no_grad():
      output = self.model(input_tensor)
      _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

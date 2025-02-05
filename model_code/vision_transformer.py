import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from timm.models.vision_transformer import VisionTransformer
import numpy as np
import cv2

# load data
def dict_to_image(image_dict):
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")

dataset_train = load_dataset('Falah/Alzheimer_MRI', split='train')
dataset_train = dataset_train.to_pandas()
dataset_train['img_arr'] = dataset_train['image'].apply(dict_to_image)
dataset_train.drop("image", axis=1, inplace=True)
dataset_test = load_dataset('Falah/Alzheimer_MRI', split='test')
dataset_test = dataset_test.to_pandas()
dataset_test['img_arr'] = dataset_test['image'].apply(dict_to_image)
dataset_test.drop("image", axis=1, inplace=True)

# Dataset class
class MRIDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_arr = self.dataframe.iloc[idx]["img_arr"]  
        label = self.dataframe.iloc[idx]["label"] 
        img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

# Hyperparameters
IMAGE_SIZE = 128
PATCH_SIZE = 8
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_SAMPLES = 1000

# datasets
train_dataset = MRIDataset(dataset_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = MRIDataset(dataset_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define transformations
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0], std=[1])
# ])

# Vision Transformer model
model = VisionTransformer(
    img_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    num_classes=NUM_CLASSES
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "C:/Users/jenni/virtualEnv/DS Capstone/DS_Capstone/model_results/vision_transformer_model.pth")

# evaluation
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n✅ Model Accuracy on Test Set: {accuracy:.2f}%")

# Run Evaluation
evaluate_model(model, test_loader, device)
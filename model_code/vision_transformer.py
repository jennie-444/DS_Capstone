import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from timm.models.vision_transformer import VisionTransformer
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve

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

# dataset class
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

# hyperparameters
IMAGE_SIZE = 128
PATCH_SIZE = 8
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

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

# vision transformer model
model = VisionTransformer(
    img_size=128,    
    patch_size=16,   
    embed_dim=384,  
    depth=8,         
    num_heads=6,   
    mlp_ratio=3.0,   
    dropout=0.1,    
    in_chans=1,      
    num_classes=NUM_CLASSES
)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# training loop
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

# save the trained model
torch.save(model.state_dict(), "C:/Users/jenni/virtualEnv/DS Capstone/DS_Capstone/model_results/vision_transformer_model_basic.pth")

# evaluation
def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) 
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    tpr = tp / (tp + fn) 
    fpr = fp / (fp + tn)  
    tnr = tn / (tn + fp)  
    fnr = fn / (fn + tp)

    # roc curve
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
    auc = roc_auc_score(all_labels_bin, np.array(all_probs), multi_class='ovr', average='macro')
    fpr_class, tpr_class, _ = roc_curve(all_labels_bin.ravel(), np.array(all_probs).ravel())
    
    # plot roc curve
    plt.figure()
    plt.plot(fpr_class, tpr_class, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("C:/Users/jenni/virtualEnv/DS Capstone/DS_Capstone/model_results/vision_transformer_roc_curve.png")
    plt.close()

    # save metrics to csv
    metrics_dict = {
        "Accuracy": [accuracy],
        "Precision": [precision],
        "F1-Score": [f1],
        "TPR (Sensitivity)": [tpr],
        "FPR": [fpr],
        "TNR (Specificity)": [tnr],
        "FNR": [fnr],
        "AUC": [auc]
    }

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv("C:/Users/jenni/virtualEnv/DS Capstone/DS_Capstone/model_results/vision_transformer_metrics_with_auc.csv", index=False)

# run evaluation
evaluate_model(model, test_loader, device)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from datasets import load_dataset
import numpy as np
import pandas as pd
import cv2
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
class ImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx]["img_arr"]
        label = self.dataframe.iloc[idx]["label"]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# CNN code
class MRI_CNN(nn.Module):
    def __init__(self, num_classes):
        super(MRI_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# hyperparameters
NUM_CLASSES = 4
N_EPOCHS = 15
BATCH_SIZE = 32

# data loader
train_dataset = ImageDataset(dataset_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = ImageDataset(dataset_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# instantiate the model
model = MRI_CNN(num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_values = []
for epoch in range(N_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_values.append(avg_loss)

    print(f"Epoch {epoch + 1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
    
# save the trained model
torch.save(model.state_dict(), "DS_Capstone/saved_models/vision_transformer_model_basic.pth")

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

    # compute confusion matrix for 4 classes
    confusion = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    
    # metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # calculate TPR, FPR, TNR, FNR for each class
    tpr_list = []
    fpr_list = []
    tnr_list = []
    fnr_list = []
    
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        tn = confusion.sum() - (tp + fn + fp)
        
        tpr = tp / (tp + fn)  
        fpr = fp / (fp + tn)  
        tnr = tn / (tn + fp)  
        fnr = fn / (fn + tp)  

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        tnr_list.append(tnr)
        fnr_list.append(fnr)

    # average TPR, FPR, TNR, FNR across all classes
    avg_tpr = np.mean(tpr_list)
    avg_fpr = np.mean(fpr_list)
    avg_tnr = np.mean(tnr_list)
    avg_fnr = np.mean(fnr_list)

    # ROC Curve and AUC
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])  
    auc = roc_auc_score(all_labels_bin, np.array(all_probs), multi_class='ovr', average='macro')
    fpr_class, tpr_class, _ = roc_curve(all_labels_bin.ravel(), np.array(all_probs).ravel())

    # plot ROC Curve
    plt.figure()
    plt.plot(fpr_class, tpr_class, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("DS_Capstone/model_results/cnn_roc_curve.png")
    plt.close()

    # save metrics to CSV
    metrics_dict = {
        "Accuracy": [accuracy],
        "Precision": [precision],
        "F1-Score": [f1],
        "Avg TPR (Sensitivity)": [avg_tpr],
        "Avg FPR": [avg_fpr],
        "Avg TNR (Specificity)": [avg_tnr],
        "Avg FNR": [avg_fnr],
        "AUC": [auc]
    }

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv("DS_Capstone/model_results/cnn_metrics.csv", index=False)

    # save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([0, 1, 2, 3])
    ax.set_yticklabels([0, 1, 2, 3])
    plt.colorbar(ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues))
    plt.tight_layout()
    plt.savefig("DS_Capstone/model_results/cnn_confusion_matrix.png")
    plt.close()

# run evaluation
evaluate_model(model, test_loader, device)

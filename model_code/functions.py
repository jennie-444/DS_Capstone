import os
import torch
from datasets import load_dataset
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
from torchvision import transforms
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from PIL import Image
import scipy.ndimage

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

# dataset class with transformations for rebalancing, data augmentation
class ImageDatasetRebalacing(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx]["img_arr"]
        label = self.dataframe.iloc[idx]["label"]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
            
        return image, label

# convert image_dict to image
def dict_to_image(image_dict):
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")

# make data loaders
def load_data(batch_size, preprocess):
    dataset_train = load_dataset('Falah/Alzheimer_MRI', split='train')
    dataset_train = dataset_train.to_pandas()
    dataset_train['img_arr'] = dataset_train['image'].apply(dict_to_image)
    dataset_train.drop("image", axis=1, inplace=True)
    dataset_test = load_dataset('Falah/Alzheimer_MRI', split='test')
    dataset_test = dataset_test.to_pandas()
    dataset_test['img_arr'] = dataset_test['image'].apply(dict_to_image)
    dataset_test.drop("image", axis=1, inplace=True)

    # preprocess
    dataset_train = preprocess_data(dataset_train, preprocess)
    dataset_test = preprocess_data(dataset_test, preprocess)

    # data loader
    train_dataset = ImageDataset(dataset_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ImageDataset(dataset_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

# load data and rebalance data
def rebalance_load_data(batch_size, preprocess):
    dataset_train = load_dataset('Falah/Alzheimer_MRI', split='train')
    dataset_train = dataset_train.to_pandas()
    dataset_train['img_arr'] = dataset_train['image'].apply(dict_to_image)
    dataset_train.drop("image", axis=1, inplace=True)
    dataset_test = load_dataset('Falah/Alzheimer_MRI', split='test')
    dataset_test = dataset_test.to_pandas()
    dataset_test['img_arr'] = dataset_test['image'].apply(dict_to_image)
    dataset_test.drop("image", axis=1, inplace=True)

    # preprocess
    dataset_train = preprocess_data(dataset_train, preprocess)
    dataset_test = preprocess_data(dataset_test, preprocess)

    test_dataset = ImageDataset(dataset_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Random transformations for data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(30),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),  
        transforms.ToTensor()
    ])

    # calculate class weights for balancing
    class_counts = np.bincount(dataset_train.label)
    class_weights = 1. / class_counts
    weights = class_weights[dataset_train.label]
    train_sampler = data.WeightedRandomSampler(weights, len(weights))

    # create datasets with transformations applied
    # for training apply data augmentation transformations
    train_dataset = ImageDatasetRebalacing(dataset_train, transform=transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        sampler=train_sampler
    )
    return train_loader, test_loader

# train function
def train_model(
    save_dir,
    model,
    train_loader,
    device,
    optimizer,
    criterion,
    num_epochs
):
    """
    Purpose: Train the model
    Args:
        save_dir: directory to save trained model weights and biases
        model: model to train
        train_loader: train data loader
        device: device to run model on
        optimizer: optimizer to use
        criterion: loss function to use
        epochs: training epoch count
    Returns: trained model
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # save model final checkpoint
    torch.save({
        'epoch': num_epochs,  
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(save_dir, 'model_checkpoint.pth'))

    return(model)

# evaluation function
def evaluate_model(model, test_loader, device, save_dir):
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
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
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
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

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
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()


# preprocessing functions

# min/max scaling
def min_max_scale(image_array, new_min=0, new_max=1):
    old_min = np.min(image_array)
    old_max = np.max(image_array)
    if old_min == old_max:
        return np.full(image_array.shape, new_min)
    scaled_array = (image_array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_array

def min_max_preprocess(dataframe):
    min_max_dataset = pd.DataFrame()
    min_max_dataset['label'] = dataframe['label']
    min_max_images = []
    for i in dataframe['img_arr']:
        scaled_image = min_max_scale(i, 0, 1)
        min_max_images.append(scaled_image)
    min_max_dataset['img_arr'] = min_max_images
    return min_max_dataset

# z score standardization
def z_score_standardize(image_array):
    mean = np.mean(image_array)
    std = np.std(image_array)
    if std == 0:
        return np.zeros_like(image_array)
    standardized_array = (image_array - mean) / std
    return standardized_array

def z_score_preprocess(dataframe):
    z_score_dataset = pd.DataFrame()
    z_score_dataset['label'] = dataframe['label']
    z_score_images = []
    for i in dataframe['img_arr']:
        scaled_image = z_score_standardize(i)
        z_score_images.append(scaled_image)
    z_score_dataset['img_arr'] = z_score_images
    return z_score_dataset

# local contrast normalization
# kernel_size (int): Size of the local neighborhood window.
# epsilon (float): Small constant to prevent division by zero.
def local_contrast_normalization(image_array, kernel_size=3, epsilon=1e-8):
    image_array = image_array.astype(np.float32)
    # Compute local mean using a uniform filter
    local_mean = scipy.ndimage.uniform_filter(image_array, size=kernel_size, mode='reflect')
    # Compute local variance using a squared filter
    local_sqr_mean = scipy.ndimage.uniform_filter(image_array ** 2, size=kernel_size, mode='reflect')
    local_std = np.sqrt(local_sqr_mean - local_mean ** 2 + epsilon)
    # Normalize the image
    normalized_image = (image_array - local_mean) / local_std
    return normalized_image

def local_contrast_preprocess(dataframe):
    local_contrast_dataset = pd.DataFrame()
    local_contrast_dataset['label'] = dataframe['label']
    local_contrast_images = []
    for i in dataframe['img_arr']:
        scaled_image = local_contrast_normalization(i)
        local_contrast_images.append(scaled_image)
    local_contrast_dataset['img_arr'] = local_contrast_images
    return local_contrast_dataset

def crop_brain_region(image):
    
    # Threshold the image to separate the brain from the background
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the non-zero regions (brain region)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("failed")
        return image

    # Get the bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Optionally draw a rectangle around the brain (for debugging purposes)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Crop the brain region to eliminate black background
    cropped = image[y:y+h, x:x+w]
    
    return cropped

def crop_preprocess(dataframe):
    cropped_dataset = pd.DataFrame()
    cropped_dataset['label'] = dataframe['label']
    cropped_images = []
    for i in dataframe['img_arr']:
        scaled_image = crop_brain_region(i)
        cropped_images.append(scaled_image)
    cropped_dataset['img_arr'] = cropped_images
    return cropped_dataset

# gaussian blur
def gaussian_blur_preprocess(dataframe):
    gaussian_dataset = pd.DataFrame()
    gaussian_dataset['label'] = dataframe['label']
    gaussian_images = []
    for i in dataframe['img_arr']:
        scaled_image = cv2.GaussianBlur(i, (5, 5), 0)
        gaussian_images.append(scaled_image)
    gaussian_dataset['img_arr'] = gaussian_images
    return gaussian_dataset

# median blur
def median_blur_preprocess(dataframe):
    median_dataset = pd.DataFrame()
    median_dataset['label'] = dataframe['label']
    median_images = []
    for i in dataframe['img_arr']:
        scaled_image = cv2.medianBlur(i, 5)
        median_images.append(scaled_image)
    median_dataset['img_arr'] = median_images
    return median_dataset

# bilateral filter
def bilateral_filer_preprocess(dataframe):
    bilateral_dataset = pd.DataFrame()
    bilateral_dataset['label'] = dataframe['label']
    bilateral_images = []
    for i in dataframe['img_arr']:
        scaled_image = cv2.bilateralFilter(i, 9, 75, 75)
        bilateral_images.append(scaled_image)
    bilateral_dataset['img_arr'] = bilateral_images
    return bilateral_dataset

def preprocess_data(dataframe, preprocess):
    if preprocess == None:
        return dataframe

    elif preprocess == 'min_max':
        return min_max_preprocess(dataframe)

    elif preprocess == 'z_score':
        return z_score_preprocess(dataframe)
    
    elif preprocess == 'local_contrast':
        return local_contrast_preprocess(dataframe)

    elif preprocess == 'crop':
        return crop_preprocess(dataframe)
    
    elif preprocess == 'gaussian_blur':
        return gaussian_blur_preprocess(dataframe)
    
    elif preprocess == 'median_blur':
        return median_blur_preprocess(dataframe)
    
    elif preprocess == 'bilateral_filter':
        return bilateral_filer_preprocess(dataframe)
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import itertools
from functions import train_model, load_data, evaluate_model, rebalance_load_data, preprocess_data

# CNN code
class MRI_CNN(nn.Module):
    '''This version of our CNN model includes Global average pooling, four convolutional layers, batch normalization
        Choices in this version
        1. Four convolutional layers helps learn more features from the images, but risks overfitting
        2. Global average pooling pools across entire feature maps to one value but not locally, which although losing spatial information, may work sufficiently for classification and avoid overfitting
        3.  Batch normalization helps standardize outputs of batches by normalizing, helps network run faster
    '''
    def __init__(self, num_classes, dropout = 0.5):
        super(MRI_CNN, self).__init__()
        
        # 4 convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Define ReLU activation function
        self.relu = nn.ReLU()
        
        # Max pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)
        
        # Dropout helps avoid overfitting
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        x = self.global_avg_pool(x) 
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# run training on CNN image classifier
def run_cnn(
    batch_size, save_dir, dropout, lr, epochs, balance, preprocess
):
    NUM_CLASSES = 4

    if balance:
        train_loader, test_loader = rebalance_load_data(batch_size, preprocess)
    else:
        train_loader, test_loader = load_data(batch_size, preprocess)
    
    # instantiate the model
    model = MRI_CNN(num_classes=NUM_CLASSES, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trained_model = train_model(save_dir, model, train_loader, device, optimizer, criterion, epochs)
    
    # run evaluation
    evaluate_model(trained_model, test_loader, device, save_dir)

# run grid search
# Note: Change values below to be desired save path, search parameters, and preprocessing technique
SAVE_PATH = "model_results/cnn_v2/cnn_median_blur/median_blur_imbalanced"
PREPROCESS = "median_blur"

# DROPOUT = [0.1, 0.3, 0.5]
# LR = [0.001, 0.0001]
# EPOCHS = [10, 50]
# BATCH_SIZE = [32]
# BALANCED = True

# best hyperparameters
DROPOUT = [0.5]
LR = [0.0001]
EPOCHS = [50]
BATCH_SIZE = [32] 
BALANCED = False

# experiment #
COUNT = 1

params = itertools.product(DROPOUT, LR, EPOCHS, BATCH_SIZE)

for i, j, k, n in params:
    # make save directory if it does not exist
    SAVE_DIR = os.path.join(SAVE_PATH, str(COUNT))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    # train and eval, save results
    run_cnn(n, SAVE_DIR, i, j, k, BALANCED, PREPROCESS)
    # save model configuration
    config = {
        "dropout": i,
        "learning_rate": j,
        "epochs": k,
        "batch_size": n
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as json_file:
        json.dump(config, json_file, indent=4)
    COUNT+=1
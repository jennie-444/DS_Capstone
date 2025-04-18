import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from timm.models.vision_transformer import VisionTransformer
import itertools
from functions import train_model, load_data, evaluate_model, rebalance_load_data

# run training on vision transformer image classifier
def run_transformer(
    batch_size, save_dir, dropout, lr, epochs, balance, preprocess
):
    # defined params
    NUM_CLASSES = 4
    IMAGE_SIZE = 128
    NUM_CLASSES = 4
    PATCH_SIZE = 16

    # balance if needed
    if balance:
        train_loader, test_loader = rebalance_load_data(batch_size, preprocess)   
    else:
        train_loader, test_loader = load_data(batch_size, preprocess)
    
    # instantiate the model
    model = VisionTransformer(
        img_size=IMAGE_SIZE,    
        patch_size=PATCH_SIZE,   
        embed_dim=384,  
        depth=8,         
        num_heads=6,   
        mlp_ratio=3.0,     
        in_chans=1,   
        drop_rate = dropout,   
        num_classes=NUM_CLASSES
    )
    # cross entropy loss and adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trained_model = train_model(save_dir, model, train_loader, device, optimizer, criterion, epochs)
    
    # run evaluation
    evaluate_model(trained_model, test_loader, device, save_dir)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# SET TRAINING PARAMETERS: MAKE MODIFICATIONS HERE

# Note: Change value below to be desired save path
SAVE_PATH = "model_results/transformer_v1/transformer_crop/crop_balanced"
# Note: Change value below to be desired preprocessing technique
# preprocess options include: 'min_max', 'z_score', 'local_contrast', 'crop', 'gaussian_blur', 'median_blur', 'bilateral_filter'
PREPROCESS = 'crop'

# Note: uncomment below to run grid search
# change params as desired
# DROPOUT = [0.1, 0.3, 0.5]
# LR = [0.001, 0.0001]
# EPOCHS = [10, 50]
# BATCH_SIZE = [32]
# BALANCED = True

# Note: comment the values below if running grid search
# change params as desired to train 1 model
# best hyperparameters (as determined via preliminary hyperparameter tuning process)
DROPOUT = [0.1]
LR = [0.0001]
EPOCHS = [50]
BATCH_SIZE = [32]
BALANCED = True
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# experiment #
COUNT = 1

# initialize iterator 
params = itertools.product(DROPOUT, LR, EPOCHS, BATCH_SIZE)

# iterator through all params and train models
for i, j, k, n in params:
    # make save directory if it does not exist
    SAVE_DIR = os.path.join(SAVE_PATH, str(COUNT))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    # train and eval, save results
    run_transformer(n, SAVE_DIR, i, j, k, BALANCED, PREPROCESS)
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
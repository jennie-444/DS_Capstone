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
    NUM_CLASSES = 4
    IMAGE_SIZE = 128
    NUM_CLASSES = 4
    PATCH_SIZE = 16

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trained_model = train_model(save_dir, model, train_loader, device, optimizer, criterion, epochs)
    
    # run evaluation
    evaluate_model(trained_model, test_loader, device, save_dir)

# run grid search 
# uncomment below to run grid search
# Note: Change values below to be desired save path and search parameters
SAVE_PATH = "model_results/transformer_v1/transformer_crop/crop_imbalanced"
PREPROCESS = 'crop'

# DROPOUT = [0.1, 0.3, 0.5]
# LR = [0.001, 0.0001]
# EPOCHS = [10, 50]
# BATCH_SIZE = [32]
# BALANCED = True

# best hyperparameters
DROPOUT = [0.1]
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
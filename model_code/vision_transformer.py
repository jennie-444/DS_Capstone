import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from timm.models.vision_transformer import VisionTransformer
from functions import train_model, load_data, evaluate_model

# run training on vision transformer image classifier
def run_transformer(
    batch_size, save_dir, dropout, lr, epochs
):
    NUM_CLASSES = 4
    IMAGE_SIZE = 128
    NUM_CLASSES = 4
    PATCH_SIZE = 16
    train_loader, test_loader = load_data(batch_size)
    
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
# Note: Change values below to be desired save path and search parameters
SAVE_PATH = "model_results/transformer"
DROPOUT = [0.1, 0.3, 0.5]
LR = [0.001, 0.0001]
EPOCHS = [10, 50]
BATCH_SIZE = [32]

# experiment #
COUNT = 1

for i in DROPOUT:
    for j in LR:
        for k in EPOCHS:
            for n in BATCH_SIZE:
                # make save directory if it does not exist
                SAVE_DIR = os.path.join(SAVE_PATH, str(COUNT))
                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)
                # train and eval, save results
                run_transformer(n, SAVE_DIR, i, j, k)
                # save model configuration
                config = {
                    "dropout": i,
                    "learning_rate": j,
                    "epochs": k,
                    "batch_size": n,
                    "balanced": False,
                    "data_preprocessing": None
                }
                with open(os.path.join(SAVE_DIR, "config.json"), "w") as json_file:
                    json.dump(config, json_file, indent=4)
                COUNT+=1
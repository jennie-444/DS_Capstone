import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
#import torchvision.transforms as transforms
from timm.models.vision_transformer import VisionTransformer
from functions import train_model, load_data, evaluate_model

# Define transformations
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0], std=[1])
# ])

# run training on vision transformer image classifier
def run_transformer(
    batch_size, save_dir, dropout, lr, epochs, patch_size
):
    NUM_CLASSES = 4
    IMAGE_SIZE = 128
    NUM_CLASSES = 4
    train_loader, test_loader = load_data(batch_size)
    
    # instantiate the model
    model = VisionTransformer(
        img_size=IMAGE_SIZE,    
        patch_size=patch_size,   
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
    evaluate_model(trained_model, test_loader, device)

# run grid search
SAVE_PATH = "model_results/transformer"
DROPOUT = [0.1, 0.3, 0.5]
LR = [0.01, 0.001, 0.0001]
EPOCHS = [10, 25, 50]
BATCH_SIZE = [32, 64, 100]
PATCH_SIZE = [16, 32]
COUNT = 1

for i in DROPOUT:
    for j in LR:
        for k in EPOCHS:
            for n in BATCH_SIZE:
                for m in PATCH_SIZE:
                    # make save directory if it does not exist
                    SAVE_DIR = os.path.join(SAVE_PATH, str(COUNT))
                    if not os.path.exists(SAVE_DIR):
                        os.makedirs(SAVE_DIR)
                    # train and eval, save results
                    run_transformer(n, SAVE_DIR, i, j, k, m)
                    # save model configuration
                    config = {
                        "dropout": i,
                        "learning_rate": j,
                        "epochs": k,
                        "batch_size": n,
                        "patch_size": m,
                        "balanced": False,
                        "data_preprocessing": None
                    }
                    with open(os.path.join(SAVE_DIR, "config.json"), "w") as json_file:
                        json.dump(config, json_file, indent=4)
                    COUNT+=1
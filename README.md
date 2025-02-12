# Alzheimer’s Disease Onset Classification From MRI Images
## Jennie An, Nadia Laswi, Jayesh Katade

### Overview

Alzheimer’s disease is the most common neurological disease due to a disorder known as cognitive impairment. The disease progresses to deterioration in cognitive abilities, behavioral changes, and memory loss. Alzheimer’s disease is considered one of the challenges facing healthcare in the modern century. There is no effective treatment to cure it, but there are drugs to slow its progression, therefore, early detection of Alzheimer’s is vital to take measures before the issue develops into brain damage that cannot be treated. MRI techniques contribute to the diagnosis and prediction of Alzheimer’s progression by detecting structural loss before cognitive decline. MRI images require highly experienced doctors and radiologists as well as a lot of time to analyze. There is a lot of potential for ML and deep learning techniques to play a vital role in analyzing huge amounts of MRI images with high accuracy to detect Alzheimer’s and predict its progression. Given this opportunity, our team has decided to build image classification models to detect Alzheimer’s progression in MRI brain scans. 

### Dataset

Ourchosen dataset is published by Hugging Face and authored by Falah.G.Salieh. This data was collected for the purpose of research of Alzheimer’s Disease. The data is publicly available, and there are no access restrictions. The dataset is linked here: https://huggingface.co/datasets/Falah/Alzheimer_MRI

#### Structure
There are just 2 columns in the dataset consisting of the image and the label associated to the image describing the severity of the disease.The image pixel array representations can be extracted. The labels for the images include non demented, moderate demented, mild demented, and very mild demented. The dataset is split into train and test sets. The training data size is 22,560,791.2 bytes and 5120 samples. The test data size if 5,637,447.08 bytes and 1280 samples. 

#### Image Preprocessing
In the `data_preprocess.ipynb` we have conducted various preprocessing steps for our dataset. Such preprocessing steps include normalization techniques such as min/max scaling, z-score standardization, and local contrast normalization. We also preformed image cropping to remove unneeded parts of the image. Lastly we performed noise reduction using methods such as gaussian blur, median blur, and bilateral filter. Any of these preprocessing functions can be called on the data before model training. 

#### Rebalancing
Our dataset contains imbalanced classes. Our class counts in the training data are 724 images for mild demented, 49 images of moderate demented, 2566 images of non-demented, and 1781 images of very mild demented. The test set is similarly distributed. To combat this imbalance we have utilized data augmentation technqiues to generate more samples in underrepresented classes. We perform random transformations including flips and rotations. 

#### CNN
`CNN.py` contains code to train a CNN for multi-class classification, conducting grid search over various hyperparameters. 

#### Vision Transformer
`vision_transformer.py` contains code to train a vision transformer for multi-class classification, conducting grid search over various hyperparameters. 

#### Repo Structure
- `data_code`: code related to preprocessing and data exploration
    - `data_load_explore.ipynb`: exploration
    - `data_preprocess.ipynb`: preprocessing functions for normalization, image cropping, and noise reduction
    - `data_rebalancing.ipynb`: rebalancing functions
- `model_code`: train and evaluation code for models
    - `CNN.py`: train, eval, grid search. Saves CNN configuration, confusion matrix, weights, metrics, ROC curve, 
    - `vision_transformer.py`: train, eval, grid search. Saves vision transformer configuration, confusion matrix, weights, metrics, ROC curve, 
- `model_results`: stores model results
    - `cnn`
    - `transformers`
- `requirements.txt`: dependencies

#### Instructions to Run Code

1. Clone the repo using `https://github.com/jennie-444/DS_Capstone.git`
2. Navigate to the project directory
3. Set up a virtual environment (optional)
4. Install the required dependencies from the requirements.txt file `pip install -r requirements.txt`
5. Optional: before running model code change the hyperparameters to conduct grid search over.
6. Optional: before running model code change the hsave directory you'd like to save your results to.
7. Run the CNN model training and evaluation using `python model_code/CNN.py`.
8. Run the transformer model training and evaluation using `python model_code/CNN.py`

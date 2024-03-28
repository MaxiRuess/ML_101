import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms,  datasets
import os
from pathlib import Path
from torchinfo import summary
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt
from tqdm import tqdm
import logging
import yaml 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    

def set_seed(seed):
    """
    This function sets the seed for reproducibility
    
    Parameters
    ----------
    seed: int
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_device():
    """
    This function sets the device to GPU if available else to CPU
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return device

def create_dataloader(train_dir, test_dir, batch_size, img_size): 
    """
    This function creates a DataLoader object for the train and test datasets
    
    Parameters
    ----------
    train_dir: str
    test_dir: str
    batch_size: int
    img_size: tuple
    num_workers: int
    """
    
    train_transforms = transforms.Compose([
        transforms.Resize(size = img_size),
        transforms.RandomHorizontalFlip(p= 0.5),
        transforms.RandomRotation(degrees = 45),
        transforms.RandomResizedCrop(size = img_size, scale = (0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(size = img_size),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.ImageFolder(root = train_dir, transform = train_transforms, target_transform = None)
    test_dataset = datasets.ImageFolder(root = test_dir, transform = test_transforms, target_transform = None)
    
    class_names = train_dataset.classes
    
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    return train_dataloader, test_dataloader, class_names


def show_img(test_dataloader, class_names): 
    """ 
    This function displays single image from a batch of images from the test dataset
    """
    img, label = next(iter(test_dataloader))
    
    random_idx = np.random.randint(img.size(0))
    img = img[random_idx]
    label = label[random_idx]
    
    plt.imshow(img.permute(1, 2, 0))
    plt.title(class_names[label.item()])
    plt.show()
    
    
class TinyVGG(nn.Module):
    
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_1 = nn.Sequential(
        nn.Conv2d(in_channels = input_shape,out_channels = hidden_units,
                  kernel_size = 2, stride = 1, padding = 0),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                  kernel_size = 2, stride = 1, padding = 0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv_2 = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                  kernel_size = 2, stride = 1, padding = 0),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                  kernel_size = 2, stride = 1, padding = 0 ),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                  kernel_size = 3, stride = 1, padding = 0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_units* 13*13, out_features = output_shape)
    )
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return self.classifier(x)


def create_summary(model: torch.nn.Module, input_size: tuple):
    """
    Create a summary of the model
    
    Args:
    model: torch.nn.Module
        Model to be summarized
    input_size: tuple
        Size of the input tensor
    device: torch.device
        Device to be used for the summary
        
    Returns:
    summary: str
        Summary of the model
    """

    return summary(model = model, 
                   input_size = input_size, 
                   col_names= ['input_size','output_size', 'num_params', 'trainable'],
                   col_width= 16,
                   row_settings= ['var_names']
                    )


def train(model, train_dataloader, test_dataloader, epochs, criterion, optimizer, device, scheduler = None):
    """
    This function trains the model
    
    Parameters
    ----------
    model: nn.Module
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    epochs: int
    criterion: nn.Module
    optimizer: nn.Module
    device: str
    """
    
    model.to(device)
    
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        
        for batch, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += (outputs.argmax(1) == labels).float().mean()
            
        scheduler.step()
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_acc / len(train_dataloader)
        
        logger.info(f'Epoch: {epoch} Training Loss: {epoch_loss}, Training Accuracy: {epoch_acc}')
        
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        
        for batch, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_acc += (outputs.argmax(1) == labels).float().mean()
            
        epoch_loss = running_loss / len(test_dataloader)
        epoch_acc = running_acc / len(test_dataloader)
        
        logger.info(f'Epoch: {epoch} Validation Loss: {epoch_loss}, Validation Accuracy: {epoch_acc}')
    
        results['train_loss'].append(running_loss)
        results['train_acc'].append(running_acc)
        results['val_loss'].append(test_loss)
        results['val_acc'].append(test_acc)
        
    return model, results


def confusion_matrix(model, test_dataloader, class_names, device):
    """
    This function creates a confusion matrix
    
    Parameters
    ----------
    model: nn.Module
    test_dataloader: DataLoader
    class_names: list 
    
    """
    
    model.eval()
    
    with torch.inference_mode():
    
        y_preds = []
        
        for inputs, labels in tqdm(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            y_preds.append(logits.argmax(1))
            
    y_preds = torch.cat(y_preds, dim=0)
    
    test_truth = torch.cat([labels for _, labels in test_dataloader], dim=0)
    confmat = ConfusionMatrix(num_classes = len(class_names), task = 'multiclass')
    
    confmat_tensor = confmat(y_preds, test_truth)
    
    fig, ax = plot_confusion_matrix(confmat_tensor.numpy(), figsize = (12, 12), class_names = class_names, show_normed = True)
    fig.show()
    
    return fig


def save_model(model, model_dir):
    """
    This function saves the model
    
    Parameters
    ----------
    model: nn.Module
    model_dir: str
    """
    
    model_path = Path(model_dir)
    model_path.mkdir(parents = True, exist_ok = True)
    model_name = "tiny_vgg.pth"
    torch.save(model.state_dict(), model_path / model_name)
    
    logger.info(f'Model saved at {model_path}')
    
    
def load_model(model, model_dir):
    """
    This function loads the model
    
    Parameters
    ----------
    model: nn.Module
    model_dir: str
    """
    
    model_path = Path(model_dir)
    
    model.load_state_dict(torch.load(model_path / 'tiny_vgg.pth'))
    
    logger.info(f'Model loaded from {model_path}')
    
    return model

# Set seeds
set_seed(42)

# Set device
try: 
    device = set_device()
    logger.info(f'Device set to {device}')
except Exception as e:
    logger.error(e)
    
# Setup the dataloaders and class names 
try:
    train_dataloader, test_dataloader, class_names = create_dataloader(train_dir = config['TINYVGG_CLF']['PARAMS']['TRAIN_DIR'],
                                                                   test_dir = config['TINYVGG_CLF']['PARAMS']['TEST_DIR'],
                                                                   batch_size= config['TINYVGG_CLF']['PARAMS']['BATCH_SIZE'],
                                                                    img_size= (64, 64))
except Exception as e:
    logger.error(f'Error in creating dataloader: {e}')

try:
    tiny_vgg = TinyVGG(input_shape = 3, hidden_units = 32, output_shape = len(class_names))
except Exception as e:
    logger.error(f'Error in creating model: {e}')

# Train model if config['TINYVGG_CLF']['TRAIN'] is True
# Otherwise go directly to loading the model
# Normally you would have a separate script for training and loading the model

if config['TINYVGG_CLF']['TRAIN']:
    try: 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(tiny_vgg.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model, results = train(model= tiny_vgg,train_dataloader= train_dataloader, test_dataloader= test_dataloader, 
            epochs= config['TINYVGG_CLF']['PARAMS']['EPOCHS'], 
            criterion= criterion, optimizer= optimizer, device= device, scheduler= scheduler)
        logger.info(f"Model training results: {results}")
        
    except Exception as e:
        logger.error(f'Error in training model: {e}')
    
    try:
        save_model(model= tiny_vgg, model_dir= config['TINYVGG_CLF']['PARAMS']['MODEL_DIR'])
    except Exception as e:
        logger.error(f'Error in saving model: {e}')
    

try: 
    tiny_vgg = load_model(model= tiny_vgg, model_dir= config['TINYVGG_CLF']['PARAMS']['MODEL_DIR'])
except Exception as e:
    logger.error(f'Error in loading model: {e}')
    


    


    
    





    

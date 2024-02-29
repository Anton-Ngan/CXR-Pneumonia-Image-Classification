import os
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from pathlib import Path
from torchmetrics.classification import BinaryAccuracy
from pneumonia_cxr_model import PneumoniaModel
from step_functions import train_model, test_step

## Device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
EPOCHS = 4
IMG_SIZE = 224



### Define the path to retrieve the dataset
data_dir = Path("data")

train_dir = data_dir/ "chest_xray/train"
test_dir = data_dir/ "chest_xray/test"


### Perform data augmentation and turn data into tensors
train_transform = transforms.Compose([
   transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
   transforms.RandomHorizontalFlip(),
   transforms.RandomVerticalFlip(),
   transforms.RandomRotation(degrees=(0, 180)),
   transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform
                                  )
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=test_transform
                                 )


# Rectify the Weight imbalance via class weighting
class_weights = []
for root, subdir, files in os.walk(train_dir):
    if len(files) > 0:
        class_weights.append(1/len(files))

sample_weights = [0] * len(train_data)
for idx, (image, label) in enumerate(train_data.imgs):
    class_weight = class_weights[label]
    sample_weights[idx] = class_weight

sampler = WeightedRandomSampler(sample_weights, 
                                num_samples=len(sample_weights),
                                replacement=True
                                )


## Turn the dataset into iterables
train_dl = DataLoader(dataset=train_data,
                      batch_size=BATCH_SIZE,
                      num_workers=NUM_WORKERS,
                      sampler=sampler
                      )

test_dl = DataLoader(dataset=test_data,
                     batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS,
                     shuffle=False
                     )


## Create an instance of the model
## Define hyperparameters, loss function and optimizer
CXRModel = PneumoniaModel(input_shape=3,
                          hidden_units=16,
                          output_shape=1
                          ).to(device)

accuracy = BinaryAccuracy().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=CXRModel.parameters(),
                            lr=0.01,
                            weight_decay=1e-2
                            )


## Create model save path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "pneumonia_cxr_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

if __name__ == "__main__":
    # Make initial inference using test dataset
    # test_loss, test_acc = test_step(model=CXRModel,
    #                                 data_loader=test_dl,
    #                                 loss_fn=loss_fn,
    #                                 device=device,
    #                                 accuracy_fn=accuracy)
    # print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc*100:.2f}%")

    ## Train the model
    model_results = train_model(model=CXRModel,
                                train_data_loader=train_dl,
                                test_data_loader=test_dl,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                epochs=EPOCHS,
                                device=device,
                                accuracy_fn=accuracy
                                )
    
    ## Save the model parameters and train results
    torch.save({"model_state_dict": CXRModel.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_fn,
                "model_results": model_results},
                f=MODEL_SAVE_PATH
                )
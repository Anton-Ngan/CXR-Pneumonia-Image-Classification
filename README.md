# CXR Pneumonia CNN

![Capture](https://github.com/Anton-Ngan/CXR-Pneumonia-Image-Classification/assets/126856263/6798c410-ac5f-42b1-acec-14afb083ebb0)

## Files
* The Jupyter Notebook (`.ipynb`) contains the script to visualise the dataset and evaluate the model
* The Python files contain the model class and the model training logic
* The folder `model_state_dict_results` contains the parameters of the trained model, model results along with the loss function and optimizer parameters
* `data/chest_xray` folder contains the dataset used to train the model

## Description
A CNN model was built and trained using the PyTorch framework to detect whether the chest x-ray (CXR) image had pneumonia or not. Pneumonia is an infection that causes 
inflammation in the lungs. AP (anterior-posterior or "front-to-back") CXRs are often taken to assist the radiologist in the diagnosis of pneumonia. Pneumonia in AP CXRs is
often characterised by radiopaque regions within the lobar region of the lungs.

The model was trained on a dataset of 5,216 images and tested on a dataset of 624 images. The CXR in the dataset was classified into two classes - normal or pneumonia. 

![Capture](https://github.com/Anton-Ngan/CXR-Pneumonia-Image-Classification/assets/126856263/360d3dfd-a000-45e9-97ae-19283137c92b)

## Major Issue 1: Imbalanced Dataset
One of the major issues in training the model is the imbalanced training dataset. The pneumonia CXRs outnumbered the normal CXRs by three fold. Class weighting was used to 
minimised the imbalance in the dataset. Without class weighting, the model would be bias towards classifying CXRs with pneumonia. This is shown in the training loop below:
* The test dataset contained 390 pneumonia CXRs out of the 624 total images in this dataset (this roughly equates to 63% of the images in test dataset).
* The train dataset contained 3,875 pneumonia CXRs out of the total 5,216 images in this dataset (this roughly equates to 74% of the images in train dataset).

![Capture](https://github.com/Anton-Ngan/CXR-Pneumonia-Image-Classification/assets/126856263/56ed8508-3dbe-44a2-8826-fed5d24698cb)


## Major Issue 2: Overfitting
Without regularisation techniques (weight decay and dropout), batch normalisation layers and data augmentation, the model only learnt the patterns of the images in the training
dataset and could not generalise to the images in the test dataset. Epochs ≥5 generally resulted in divergence between the training and test loss, which is an indication of overfitting.

![1](https://github.com/Anton-Ngan/CXR-Pneumonia-Image-Classification/assets/126856263/8d84d634-3f9b-47b1-922b-1765072c28c4)
![423454445_765211681912535_4810372728820301335_n](https://github.com/Anton-Ngan/CXR-Pneumonia-Image-Classification/assets/126856263/aec1c871-586e-4173-b926-f2a9e88d5488)

## Model
A CNN model with 24-layers was built and trained to detect pneumonia in the CXRs. 
* Batch normalisation layers were added to the model to help reduce the model from relying heavily on specific neurons
* Dropout layer was added to also prevent the model from overfitting
* Multiple convolution layers were added to the model to extract the prominent features in the CXRs

The model was trained under 4 epochs.
* The loss function used was binary cross entropy
* The optimizer was SGD

![Capture](https://github.com/Anton-Ngan/CXR-Pneumonia-Image-Classification/assets/126856263/e2587c17-15ea-4a46-bc4a-6f60a6a4f8c5)

## Model Results and Evaluation
An inference was performed on the test dataset before the model was trained. After one epoch, the train loss was significantly reduced by approximately 0.35 
and the accuracy rose from roughly 36% to 85%. After the fourth epoch, the training loss was approximately 0.33 and the test accuracy was approximately 87%.

![image](https://github.com/Anton-Ngan/CXR-Pneumonia-Image-Classification/assets/126856263/f6004333-8e5d-4d10-9c7f-110ec64aaa2e)


The confusion matrix shows that the model is more prone to type II errors (as it has 55 cases where it detected pneumonia in normal (truth) CXR). In a medical imaging context,
type II errors would be much preferred over type I errors. This is because failure to diagnose a patient with an illness is much more fatal than 
diagnosing a patient without an illness. However, this does not mean that type I errors should be relaxed. The MRPBA recognises that radiographers should 
prevent unnecessary radiation exposure to the patients. 

![image](https://github.com/Anton-Ngan/CXR-Pneumonia-Image-Classification/assets/126856263/4a35cfb1-16be-4c93-b187-d1db5173965e)


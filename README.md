# Dental-Disease-X-Ray-Classifier-CNN-PyTorch

## Overview:
The goal of this model is to be able to be inputted an x-ray image from a patient with a root dental disease and able to classify what disease it is, thus automating and improving the treatment of diseases as such.
This can hopefully in the futureimprove diagnosis' in the dental field for more efficient and effective treatments. This model was built using PyTorch as it is one of the best and most versatile frameworks for deep 
learning tasks. It yields an approximate of 80% accuracy over all 7 of the diseases it was trained to classify.

## Data:
- *Kaggle: https://www.kaggle.com/datasets/engineeringubu/root-disease-x-ray-imaging*
- X-Ray images from different angles of patients with 7 different root dental diseases:
  1. Irreversible pulpitis with Acute periodontitis
  2. Impacted tooth (fully bony impaction)
  3. Improper restoration with chronic apical periodontitis
  4. Chronic apical periodontitis with vertical bone loss
  5. Embedded tooth
  6. Dental caries (proximal)
  7. Periodontitis
- The images in the data differ depending on the disease and picture so all images used for training had to be altered to 500x500 pixels as that was about the median
- Used "TD-1" for training and validation and "TD-2" for predicting the diseases on the trained model

## Model Architecture:
*The deep learning model is a Convolutional Neural Network with 2 two fully connected layers and 2 convolutional layers. The input shape is 3 units, it has 7 hidden units, and an output shape of 7
because of number of classes. The architecture may be changed later to optimize results.* Below is a basic diagram of a convolutional neural network simlar to this.


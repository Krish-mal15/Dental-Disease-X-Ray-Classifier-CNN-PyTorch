# Dental-Disease-X-Ray-Classifier-CNN-PyTorch


## Overview:
The goal of this model is to be able to be inputted an x-ray image from a patient with a root dental disease and able to classify what disease it is, thus automating and improving the treatment of diseases as such.
This can hopefully in the futureimprove diagnosis' in the dental field for more efficient and effective treatments. This model was built using PyTorch as it is one of the best and most versatile frameworks for deep 
learning tasks. It yields an approximate of 80% accuracy over all 7 of the diseases it was trained to classify. In summary, the model extracts distinct features from the images and learns them to further classify other images similar to the ones it learned to make a prediction on an input imaging, hence allowing it to classify what disease the image belongs to.


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
- To get image into desired format, I used torchvisions transform library to convert it into a tensor and normalize it.


## Model Architecture:
*The deep learning model is a Convolutional Neural Network with 2 two fully connected layers and 2 convolutional layers. The input shape is 3 units, it has 7 hidden units, and an output shape of 7
because of number of classes. The architecture may be changed later to optimize results.* Below is a basic diagram of a convolutional neural network simlar to this.


![Dental Disease Classifier - PyTorch Convolutional Neural Network](https://github.com/Krish-mal15/Dental-Disease-X-Ray-Classifier-CNN-PyTorch/assets/156712020/69011b73-d2ea-4958-8588-134c81b4eca2)

This model, like many other CNNs, uses 2 convolutional layers in each was two convolutional blocks which after each was a ReLU activation function (shown by diagram) and after both blocks, I used Max Pooling to retrive the highest value pixel in the image. The optimizer I used was Stochastic Gradient Descent (SGD). This proved most effective for my application. The image below diagrams the SGD algorithm (credit: https://medium.com/analytics-vidhya/stochastic-gradient-descent-1ab661fabf89)

![sgd](https://github.com/Krish-mal15/Dental-Disease-X-Ray-Classifier-CNN-PyTorch/assets/156712020/34595778-4854-49a5-88c9-6d164bcf0daa)

I used the cross-entropy loss function for my model as it is a basic loss function and yields the best results for multi-class classification. Furthermore, after I built my basic model architecture, I added the classifier section to the "DentalModel" class. In this, I faced some issues regarding the input shape and realizing it must be 3 as the coloring on the dataset wasn't consistent. I then had to calculate the convolutional output shape to be able to input that into the Linear portion of the classifier. Additionally, I decided not to use a softmax activation function as results came out fine, but I may change this in the future


## Implementation:
- Afted I added my training loop, I used the in-built PyTorch funciton to save my model as a .pth file. In this file, the weights taht contribute to the classifier equation are stored (as shown in the bottom right of the top architecture diagram)
- I then made a new file and loaded my model, settting it into evaluation mode
- Moreover, I transformed the input test image the same way I transformed the traning images. The test data was from a separate file with similar images.
- I inputted the test data into the model and printd out the predicted class
- Lastly, I made a funciton to print out the disease name associated with the class numbers [0-6]




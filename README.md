# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

To classify the given images into it's category
## Neural Network Model

![423398622-acb92196-8a29-40c9-9505-82bf31af77b3](https://github.com/user-attachments/assets/8325b12f-5924-4c4b-a107-2fc485a2731e)


## DESIGN STEPS

## STEP 1:
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

## STEP 2:
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

## STEP 3:
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

## STEP 4:
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

## STEP 5:
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

## STEP 5:
Save the trained model, visualize predictions, and integrate it into an application if needed.
## PROGRAM

### Name:PRIYADHARSHINI.P
### Register Number:212223240128
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: Priyadharshini.P')
        print('Register Number: 212223240128')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch
![image](https://github.com/user-attachments/assets/fd786583-0102-4333-8a73-f786320d52eb)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/84743587-d491-4905-9add-e314bc1addee)



### Classification Report
![image](https://github.com/user-attachments/assets/0ef1c007-a5a9-4b75-b848-eadc88b791a3)




### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/d4166861-d6ec-4b41-91aa-0034c2b2b9a6)


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.

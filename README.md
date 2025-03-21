# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Regression is a machine learning task used to predict continuous numerical values. In this experiment, we aim to develop a simple neural network regression model with a single-layer architecture. The model takes input features, processes them using weighted connections and an activation function, and produces a continuous output. It optimizes weights using backpropagation and gradient descent to minimize error and improve prediction accuracy.

## Neural Network Model

![image](https://github.com/user-attachments/assets/43ae2420-4d26-4d34-b540-24f182234480)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects, fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

### Name: DHARSHINI K
### Register Number: 212223230047
```python
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn. Linear (1, 12)
    self.fc2 = nn. Linear (12, 10)
    self.fc3 = nn. Linear (10, 1)
    self.relu = nn.ReLU()
    self.history = {'loss': []}

  def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.fc3(x) # No activation here since it's a regression task
      return x

# Initialize the Model, Loss Function, and Optimizer

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001);

def train_model (ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion (ai_brain (X_train), y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

## Dataset Information

![image](https://github.com/user-attachments/assets/5b96479e-e1eb-4625-85d0-e01a79feec68)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/12246b60-e8ca-45c6-bc55-ba5d8a209a82)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/6dcb6a3c-ed35-4edc-b578-2759c1428b37)

## RESULT

The neural network regression model was successfully developed and trained. The model effectively minimized loss over iterations, and predictions on new sample data were accurate.

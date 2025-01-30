# Sonar Rock vs. Mine Prediction

## Overview
This repository contains a machine learning model to classify objects as either rocks or mines using sonar data. The model is built using logistic regression and evaluated based on accuracy.

## Dependencies
Before running the code, ensure you have the following dependencies installed:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Data Collection and Processing
The dataset used is the Sonar dataset, which contains 208 samples with 60 features each. The last column represents the label ('R' for rock, 'M' for mine).

### Loading the Dataset
```python
# Loading the dataset
df = pd.read_csv('/content/Copy of sonar data.csv', header=None)
```

### Displaying the First Few Rows
```python
df.head()
```

### Statistical Measures of the Data
```python
df.describe()
```

## Model Training
1. Split the dataset into training and testing sets.
2. Train a logistic regression model.
3. Evaluate the model's accuracy.

### Splitting the Dataset
```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Training the Model
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Model Evaluation
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

## Results
- The accuracy of the logistic regression model on the test data will be displayed.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sonar-rock-mine-classification.git
   ```
2. Install dependencies using pip:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Run the Python script:
   ```bash
   python sonar_classification.py
   ```

## License
This project is licensed under Siddharthan Channel [YouTube] Machine Learning. link : [https://youtu.be/fiz1ORTBGpY?si=4oCP_4KPzqNlD5vm]


# Logistic Regression Project

Welcome to my logistic regression project! 🚀 In this repository, you'll find my Python implementation of a logistic regression model. I've covered everything from data preprocessing to model evaluation using various metrics.

## Installation

Make sure to install the necessary Python packages before running the code. You can do this by using the following command:

```bash
%pip install scikit-learn pandas numpy matplotlib pylab scipy
```

## Imports

Take a look at the Python libraries I've used in the code:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pylab as pl
import scipy.optimize as opt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import log_loss
```

## Workflow

Here's a quick rundown of what happens in the code:

1. **Data Preprocessing**: I clean and preprocess the data.
2. **Normalize Data**: Data normalization is crucial for optimal model performance.
3. **Train-Test Split**: I split the data into training and testing sets.
4. **Modeling with Logistic Regression**: The magic happens as I train a logistic regression model.
5. **Evaluation**: Dive into the Jaccard index, confusion matrix, and log loss to gauge the model's performance.

Feel free to explore the code for a more in-depth understanding. Happy coding! 🤖💻

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
print("First 10 rows of the dataset:\n", data.head(10))

print("\nDataset Summary:")
print("Number of instances:", data.shape[0])
print("Number of features:", data.shape[1] - 1) 
print("Number of target classes:", len(np.unique(data['species'])))
print("\nFeature description:")
for column in data.columns[:-1]:
    print(f"{column}: Type={data[column].dtype}, Min={data[column].min()}, Max={data[column].max()}")

 
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap of Feature Correlations")
plt.show()

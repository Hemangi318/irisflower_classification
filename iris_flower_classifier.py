import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df=pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\New Folder\\IRIS.csv")
df
df.isna().sum()
df.info()
df.describe()
df.head()
df['species'].unique()
duplicates=df.duplicated()
df[duplicates]
df=df.drop_duplicates()
df=df.reset_index()
df['sepal_length'].unique()
df['sepal_width'].unique()
df['petal_length'].unique()
df['petal_width'].unique()
df["species"].value_counts()
sns.pairplot(df, hue="species")
plt.show()
df_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for col in df_cols:
    sns.boxplot(x="species",y=col,data=df) 
    plt.show()
    le = LabelEncoder()
df["species"] = le.fit_transform(df['species'])
df.columns
df.drop(columns=['index'],inplace=True)
sns.heatmap(df.corr(),annot=True)
X=df.drop(columns=["species"])
y=df["species"]
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42, stratify = y)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "KNN": make_pipeline(StandardScaler(),KNeighborsClassifier()),
    "SVM": make_pipeline(StandardScaler(),SVC())
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {acc:.3f}")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nConfusion Matrix for the {name} model:\n{confusion_matrix(y_test, y_pred)}\n")
    print(f"Classification Report for the {name} model:\n{classification_report(y_test, y_pred)}")
    print(f"{name} model's Accuracy: {acc:.3f}\n{'-'*70}")
    svm_pipeline=make_pipeline(StandardScaler(),SVC())
svm_pipeline.fit(X_train,y_train)
predictions=svm_pipeline.predict(X_test)
compare_df = pd.DataFrame({'actual': y_test, 'predicted': predictions})
compare_df = compare_df.reset_index(drop = True)
compare_df
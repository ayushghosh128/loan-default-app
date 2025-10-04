"""
Loan Default Prediction - Simplified Decision Tree Version
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
df = pd.read_csv("../data/hmeq.csv")

# 2. Handle Missing Values
num_cols = df.select_dtypes(include=['float64','int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# 3. Encode Categoricals
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4. Split Data
X = df.drop("BAD", axis=1)
y = df["BAD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 5. Train Decision Tree Model
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
tree.fit(X_train, y_train)

# 6. Evaluate
y_pred = tree.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Visualize Tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=["Good","Bad"], filled=True)
plt.title("Decision Tree for Loan Default Prediction")
os.makedirs("../outputs", exist_ok=True)
plt.savefig("../outputs/tree_plot.png", dpi=300)
plt.show()

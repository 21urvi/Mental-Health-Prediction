import numpy as np
import pandas as pd
import pickle
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("mental_health.csv")

# Define gender categories
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]

# Normalize Gender column
data['Gender'] = data['Gender'].astype(str).str.lower()
data['Gender'] = data['Gender'].apply(
    lambda x: 'male' if x in male_str else ('female' if x in female_str else ('trans' if x in trans_str else np.nan))
)

# Remove unwanted gender values
data = data.dropna(subset=['Gender'])  # Remove rows where Gender is NaN

# Encode categorical data
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1, 'trans': 2})
data['family_history'] = data['family_history'].map({'No': 0, 'Yes': 1})
data['treatment'] = data['treatment'].map({'No': 0, 'Yes': 1})

# Ensure all columns are numeric
data = data.dropna()  # Remove rows with missing values
data = data.apply(pd.to_numeric)  # Convert all columns to numeric

# Split data into features (X) and labels (y)
X = data.drop(columns=["treatment"]).values  # Replace "treatment" with actual target column name
y = data["treatment"].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define classifiers
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()

# Create Stacking Classifier (using sklearn's version)
stack = StackingClassifier(estimators=[('knn', clf1), ('rf', clf2), ('nb', clf3)], final_estimator=lr)

# Train the model
stack.fit(X_train, y_train)

# Save model using pickle
with open("model.pkl", "wb") as f:
    pickle.dump(stack, f)

print("✅ Model saved as 'model.pkl'")

# Load model to test if it saved correctly
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Test with a sample input
sample_input = np.array([25, 1, 0]).reshape(1, -1)  # Adjust features based on dataset
prediction = model.predict(sample_input)
print(f"Sample Prediction: {prediction}")

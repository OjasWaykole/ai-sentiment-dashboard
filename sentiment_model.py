import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\DOCS\train.csv', encoding='latin-1')

# Select required columns
data = data[['text','sentiment']]

# REMOVE EMPTY TEXT ROWS
data = data.dropna()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], test_size=0.2
)

# Convert text to numbers
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

model.fit(X_train_vec, y_train)

# Test model
predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


# Test with custom input
user_input = input("Enter a sentence: ")

user_vec = vectorizer.transform([user_input])

prediction = model.predict(user_vec)

print("Predicted Sentiment:", prediction[0])
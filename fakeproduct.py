import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Sample dataset containing product descriptions and their labels (1 for fake, 0 for real)
data = {
    'product_description': [
        'This is a brand new iPhone 12, best price online!',
        'Buy luxury watches for 50% off, limited time offer.',
        'Genuine Apple product, comes with warranty.',
        'This handbag is made with authentic leather from Italy.',
        'High-quality fake Rolex watch at an affordable price.',
        'Brand new Samsung Galaxy with all accessories included.',
        '100% authentic Gucci bag, limited edition!',
        'Super cheap electronics, not recommended!'
    ],
    'label': [0, 1, 0, 0, 1, 0, 0, 1]  # 0 = Real, 1 = Fake
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into features (X) and target (y)
X = df['product_description']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numeric using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('product_faker_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Example of prediction for a new product description
new_description = ["Exclusive authentic sneakers with original packaging."]
vectorized_description = vectorizer.transform(new_description)
prediction = classifier.predict(vectorized_description)
print(f"Prediction (0 = Real, 1 = Fake): {prediction[0]}")

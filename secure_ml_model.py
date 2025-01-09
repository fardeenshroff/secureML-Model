# secure_ml_model.py

from cryptography.fernet import Fernet
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Function to generate encryption key
def generate_key():
    return Fernet.generate_key()

# Function to encrypt data
def encrypt_data(data, key):
    cipher_suite = Fernet(key)
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data

# Function to decrypt data
def decrypt_data(encrypted_data, key):
    cipher_suite = Fernet(key)
    decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
    return decrypted_data

# Example dataset (you can replace this with your real dataset)
data = {'Feature1': [1, 2, 3, 4],
        'Feature2': [10, 20, 30, 40],
        'Label': [0, 1, 0, 1]}
df = pd.DataFrame(data)

# Encrypt the data before training
key = generate_key()
df['Feature1'] = df['Feature1'].apply(lambda x: encrypt_data(str(x), key))
df['Feature2'] = df['Feature2'].apply(lambda x: encrypt_data(str(x), key))

# Decrypt the data before using it for model training
df['Feature1'] = df['Feature1'].apply(lambda x: decrypt_data(x, key))
df['Feature2'] = df['Feature2'].apply(lambda x: decrypt_data(x, key))

# Split data into features and labels
X = df[['Feature1', 'Feature2']].astype(float)
y = df['Label']

# Initialize and train a machine learning model
model = RandomForestClassifier()
model.fit(X, y)

# Model prediction (example)
prediction = model.predict([[3, 30]])
print(f'Prediction: {prediction}')

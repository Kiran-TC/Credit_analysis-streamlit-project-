💳 Credit Card Fraud Detection
A Streamlit web app that detects fraudulent credit card transactions using Logistic Regression. This project demonstrates a full data science workflow including data preprocessing, model training, and performance evaluation.

🔍 Features
Uploads and processes real-world credit card transaction data.

Trains a Logistic Regression model with scaled features.

Displays accuracy, confusion matrix, and classification report.

Interactive interface using Streamlit for real-time exploration.

🛠 Tech Stack
Python

Pandas, NumPy, Scikit-learn – for data manipulation and modeling

Streamlit – for building the interactive web app

Matplotlib/Seaborn (optional) – for extended visualizations

🚀 How to Run
Clone the repo or copy the app.py file.

Ensure the creditcard.csv dataset is in the same directory.

Run the app:

bash
Copy
Edit
pip install -r requirements.txt
streamlit run app.py
📊 Dataset
Source: Kaggle Credit Card Fraud Dataset

Contains 284,807 transactions, with Class = 1 indicating fraud.

📈 Model Performance
Logistic Regression trained with scaled features.

Achieved ~99.2% accuracy on test data.

Evaluation metrics include confusion matrix and precision/recall report.


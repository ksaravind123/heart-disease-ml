# Heart Disease Prediction Using Machine Learning

This project predicts the likelihood of heart disease using machine learning models trained on clinical patient data. It uses algorithms such as Logistic Regression, Decision Tree, Random Forest, KNN, and SVM, and selects the best-performing model for deployment.

## 🚀 Features
- Data preprocessing and encoding  
- Multiple ML model training and comparison  
- Performance evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  
- Streamlit web app for real-time predictions  
- Saved model using Joblib  

## 📊 Dataset
The dataset contains clinical attributes such as:
- Age, Sex, Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Serum Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG results
- Maximum Heart Rate (thalach)
- Exercise-Induced Angina (exang)
- ST Depression (oldpeak)
- Slope, CA, Thal
- Condition (0 = No disease, 1 = Disease)

## 🛠 Technologies Used
- Python  
- Scikit-learn  
- Pandas & NumPy  
- Streamlit  
- Matplotlib / Seaborn  
- Joblib  

## 📦 Installation
```bash
git clone https://github.com/ksaravind123/heart-disease-ml.git
cd heart-disease-ml
pip install -r requirements.txt

Run the Model Training
python notebook_train.py

Run the Web App
streamlit run app_streamlit.py



cd heart-disease-ml
pip install -r requirements.txt

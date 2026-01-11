# Hotel Booking Cancellation Prediction

This project predicts whether a hotel booking will be **canceled or not** using machine learning techniques.  
The goal is to help hotels understand booking behavior and reduce revenue loss due to cancellations.

---

## 📌 Problem Statement
Hotel booking cancellations cause significant revenue loss.  
This project builds a machine learning model to predict booking cancellations **before the arrival date** using customer and booking-related information.

---

## 📊 Dataset
- Hotel booking dataset (City Hotel & Resort Hotel)
- Includes booking details such as:
  - Lead time
  - Number of adults, children, babies
  - Meal type
  - Market segment
  - Distribution channel
  - ADR (Average Daily Rate)
  - Booking changes, etc.

---

## 🧹 Data Preprocessing
The following steps were applied:
- Removed duplicate rows
- Handled missing values
- Removed invalid bookings (e.g., adults = 0)
- Dropped post-booking leakage columns:
  - `reservation_status`
  - `reservation_status_date`
- Encoded categorical variables
- Feature scaling using `StandardScaler`

---

## 🤖 Machine Learning Models Used
- **Logistic Regression** (baseline model)
- **Random Forest Classifier** (best performing)
- **Gradient Boosting Classifier**

👉 **Final model selected:** Random Forest

---

## 📈 Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Random Forest showed the best balance between performance and robustness.

---

## 💾 Model Saving
The trained components were saved using Pickle.

⚠️ **Important Note:**  
The trained model files (`rf_model.pkl`, `scaler.pkl`, `model_columns.pkl` ) is **not included** in this repository due to GitHub file size limitations.  
Run the notebook to train the model and generate the `.pkl` file locally.

---

## 🚀 Deployment Ready
After training, the generated `.pkl` files can be used for:
- Streamlit web application
- Flask / FastAPI API
- Future predictions without retraining

---

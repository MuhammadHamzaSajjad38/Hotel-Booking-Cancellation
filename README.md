# Hotel-Booking-Cancellation

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
  - Number of adults, children
  - Meal type
  - Market segment
  - ADR (Average Daily Rate)
  - Booking changes, etc.

---

## 🧹 Data Preprocessing
The following steps were applied:
- Removed duplicate rows
- Handled missing values
- Removed invalid bookings (e.g., adults = 0)
- Dropped leakage columns:
  - `reservation_status`
  - `reservation_status_date`
- Encoded categorical variables
- Feature scaling using `StandardScaler`

---

## 🤖 Machine Learning Models Used
- **Logistic Regression** (baseline model)
- **Random Forest Classifier** (best performing)
- **Gradient Boosting Classifier**

👉 Final model selected: **Random Forest**

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
The trained components were saved using Pickle:
- `rf_model.pkl` → Trained Random Forest model
- `scaler.pkl` → Feature scaler
- `model_columns.pkl` → Feature column order (important for deployment)

---

## 🚀 Deployment Ready
The saved `.pkl` files can be used for:
- Streamlit web app
- Flask / FastAPI API
- Future predictions without retraining

---

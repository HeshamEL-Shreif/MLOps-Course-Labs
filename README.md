# Churn Prediction with MLflow Logging & Monitoring

This project demonstrates an end-to-end MLOps workflow using **MLflow** for tracking and monitoring experiments. The focus is on **model development**, **experiment tracking**, and **performance monitoring** using **Logistic Regression** and **Support Vector Machine (SVM)** models from `scikit-learn`.

---

## 🧱 Project Structure
```text
.
├── data/                      # Dataset and data handler
│   ├── Churn_Modelling.csv    # Raw data
│   └── data_handler.py        # Data preprocessing logic
│
├── mlartifacts/              # Saved MLflow models/artifacts
│
├── mlruns/                   # MLflow experiment run logs
│
├── models/                   # Model definitions
│   ├── logistic_regression.py
│   └── svm.py
│
├── monitoring_logging/       # Logging and monitoring utilities
│   ├── logging_setup.py
│   └── monitoring.py
│
├── src/                      # Core training & inference logic
│   ├── train.py
│   ├── inference.py
│   └── confusion_matrix.png
│
├── main.py                   # Entry point to run training/testing
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```
---

## ⚙️ Features

- 📊 **Experiment Tracking** with [MLflow](https://mlflow.org)
- 🧠 **Logistic Regression** and **SVM** with `scikit-learn`
- 📈 **Model Evaluation** with metrics and confusion matrix
- 🔧 **Modular Code Structure** for maintainability
- 🔍 **Custom Logging** using Python’s `logging` module

---

## 🚀 How to Run

### 1. 📦 Install Dependencies
```bash
pip install -r requirements.txt
```
2. ▶️ Run the Project
```bash
python main.py
```

This will:
- Load and preprocess the churn dataset
- Train and evaluate Logistic Regression and SVM models
- Log parameters, metrics, and artifacts to MLflow

3. 📡 Launch the MLflow UI
```bash
mlflow ui
```
Open http://127.0.0.1:5-  in your browser to view experiment logs.

⸻

🧪 Experimentation

You can tune the models by editing:
- main.py

Each run logs:
- Model parameters
- Accuracy, MSE
- Confusion matrix (as image)
- Serialized model artifact

⸻

📁 Key Modules
## 📁 Key Modules


| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `data_handler.py`       | Loads and cleans the churn dataset                   |
| `logistic_regression.py` | Builds and returns a logistic regression model |
| `svm.py`              | Builds and returns a support vector machine model|
| `train.py`            | Trains models and logs with MLflow               |
| `inference.py`        | Runs model inference                             |
| `monitoring.py`       | Evaluates and logs metrics                       |
| `logging_setup.py`    | Configures project-level logging                 |


📷 Sample Output

The confusion_matrix.png is saved and logged for each run in MLflow.

⸻

📌 Dependencies

See requirements.txt for the full list, including:
- scikit-learn
- pandas
- mlflow
- matplotlib

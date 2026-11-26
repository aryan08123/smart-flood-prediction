# 🌊 Smart Disaster Prediction – Flood Prediction Using Machine Learning

A Machine Learning project that analyzes historical rainfall data to estimate flood risk using historical rainfall patterns and classification algorithms.

## 📌 Project Overview

Floods are among the most destructive natural disasters, causing significant damage to human life, infrastructure, agriculture, and the environment.

The objective of this project is to develop a machine learning-based flood prediction system using historical rainfall data.

The project explores multiple classification algorithms, performs feature engineering, handles data preprocessing, evaluates model performance, and identifies important rainfall features associated with flood classification.

## 🎯 Objectives

- Analyze historical rainfall data using Exploratory Data Analysis (EDA).
- Perform data cleaning and feature engineering.
- Create historical rainfall-based features using lag techniques.
- Build machine learning classification models.
- Compare Logistic Regression, Random Forest, and XGBoost.
- Perform hyperparameter tuning to improve model performance.
- Evaluate models using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- Identify important rainfall features influencing model predictions.
- Develop a flood prediction interface in future project stages.

## 📂 Dataset

**Dataset:** Monthly Rainfall – Flood Data (1901–2015)

The dataset contains historical rainfall information across different subdivisions of India.

Important data attributes include:

- YEAR
- SUBDIVISION
- Monthly rainfall values (January–December)
- ANNUAL
- MONSOON_RAINFALL
- MAX_MONTHLY_RAINFALL
- AVG_MONTHLY_RAINFALL
- RAINFALL_STD
- Flood (Target Variable)

**Dataset size:** 4,090 records before lag-feature processing.

**Target variable:**
- `0` – No Flood
- `1` – Flood

**Important dataset limitation:** The Flood label in this dataset is determined by an annual rainfall threshold of 3000 mm. Therefore, it represents a rainfall-threshold-based classification rather than independently verified real-world flood occurrences.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| XGBoost | Gradient boosting classification |
| Joblib | Model serialization |

## ⚙️ Project Workflow

```text
Historical Rainfall Dataset
          |
          v
     Data Loading
          |
          v
 Exploratory Data Analysis
          |
          v
    Data Cleaning
          |
          v
  Feature Engineering
          |
          v
    Preprocessing
          |
          v
 Historical Lag Features
          |
          v
 Time-Based Train/Test Split
          |
          v
   Model Development
          |
          +----------------+
          |                |
          v                v
 Logistic Regression  Random Forest
          |                |
          +--------+-------+
                   |
                   v
                XGBoost
                   |
                   v
       Hyperparameter Tuning
                   |
                   v
         Model Evaluation
                   |
                   v
       Feature Importance
                   |
                   v
          Final Prediction
```

## 🧠 Machine Learning Algorithms

### 1. Logistic Regression

Used as the baseline classification model to establish initial performance.

### 2. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

### 3. XGBoost Classifier

A gradient boosting algorithm that builds decision trees sequentially to improve prediction performance.

### 4. Hyperparameter Tuning

GridSearchCV was used to identify better XGBoost configurations.

Selected parameters:

```python
{
    "learning_rate": 0.03,
    "max_depth": 5,
    "n_estimators": 200
}
```

## 📊 Model Performance

The following results were obtained using the time-based test dataset.

| Evaluation Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 95.60% | 95.24% | 95.36% |
| Precision | 76.19% | 69.39% | 70.83% |
| Recall | 55.17% | 58.62% | 58.62% |
| F1 Score | 64.00% | 63.55% | 64.15% |

### Tuned XGBoost Performance

| Metric | Result |
|---|---:|
| Accuracy | 95.60% |
| Precision | 71.15% |
| Recall | 63.79% |
| F1 Score | 67.27% |
| True Positives | 37 |
| True Negatives | 746 |
| False Positives | 15 |
| False Negatives | 21 |

**Current best model:** Tuned XGBoost, based on its F1-score and recall among the evaluated models.

Note: These are preliminary results from the current experimental setup. They should not be interpreted as verified real-world flood forecasting accuracy.

## 🔍 Feature Engineering

To reduce target leakage, historical rainfall features were created using the previous year's rainfall information.

The following lag features were created:

```python
PREV_ANNUAL_RAINFALL
PREV_MONSOON_RAINFALL
PREV_MAX_MONTHLY_RAINFALL
PREV_RAINFALL_STD
```

These features allow the model to use previous-year rainfall information to classify the following year's rainfall-threshold-based flood label.

## 📈 Feature Importance

The tuned XGBoost model identified the following feature importance values:

| Feature | Importance |
|---|---:|
| PREV_ANNUAL_RAINFALL | 0.726759 |
| PREV_MAX_MONTHLY_RAINFALL | 0.129572 |
| PREV_RAINFALL_STD | 0.091829 |
| PREV_MONSOON_RAINFALL | 0.051849 |

Previous annual rainfall was the most influential feature in the model's predictions.

Feature importance indicates model reliance, not causation.

## 📏 Evaluation Metrics

The project uses the following evaluation metrics:

- **Accuracy:** Measures the overall proportion of correct predictions.
- **Precision:** Measures how many predicted flood cases were actually positive.
- **Recall:** Measures how many actual positive cases were detected.
- **F1 Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Identifies true positives, true negatives, false positives, and false negatives.

Recall is particularly important because missed positive cases are an important concern in flood-risk applications.

## 🚀 Running the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project folder

```bash
cd Smart-Disaster-Prediction
```

### Step 3: Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### Step 4: Open Google Colab

Upload the project notebooks and dataset to Google Drive.

### Step 5: Execute notebooks in order

1. EDA Notebook
2. Feature Engineering Notebook
3. Preprocessing Notebook
4. Model Training Notebook

Execute the cells sequentially.

## 📁 Project Structure

```text
Smart-Disaster-Prediction/
│
├── Dataset/
│   └── Monthly Rainfall - flood Data 1901-2015.csv
│
├── Notebooks/
│   ├── Day_1_2_EDA.ipynb
│   ├── Day_3_Feature_Engineering.ipynb
│   ├── Day_4_Preprocessing.ipynb
│   └── Day_5_Model_Training.ipynb
│
├── Models/
│   └── Tuned_XGBoost_Model.pkl
│
├── Visualizations/
│
├── README.md
│
└── requirements.txt
```

*Note: The model, visualization, and deployment files are planned project structure; add them as they are created.*

## 🔮 Future Improvements

- Complete feature importance visualization.
- Perform prediction probability and threshold analysis.
- Generate ROC and Precision-Recall curves.
- Improve flood detection recall.
- Save and deploy the trained machine learning model.
- Develop an interactive flood prediction dashboard.
- Integrate additional weather and environmental information.
- Validate predictions against independently observed flood-event data.
- Explore location-specific and multi-year flood-risk forecasting.

## 🎓 Project Learning Outcomes

Through this project, the following concepts are explored:

- Exploratory Data Analysis
- Data Cleaning and Preprocessing
- Feature Engineering
- Handling Imbalanced Datasets
- Logistic Regression
- Random Forest
- XGBoost
- Hyperparameter Optimization
- Time-Based Train/Test Splitting
- Model Evaluation
- Feature Importance
- Machine Learning Model Deployment

## 👨‍💻 Author

**Aryan Sinha**

Machine Learning | Data Science | Python

## 📜 Disclaimer

This project is developed for academic and educational purposes. Its predictions are based on historical rainfall data and a rainfall-threshold-derived target label. It is not an operational flood early-warning system and should not be used for emergency decisions.

---

⭐ If you find this project interesting, consider starring the repository!

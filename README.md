# Kidney Disease Machine Learning Analysis

## 📌 Project Overview

This project applies **Machine Learning techniques to a kidney disease dataset** to explore patient-related features and build predictive and unsupervised learning models.

The project is divided into two major parts:

1. **Classification** – Predicting `CKD_Status` using multiple supervised machine learning algorithms.
2. **Clustering** – Identifying groups/patterns within the patient data using unsupervised machine learning techniques.

The analysis is implemented using **Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn**, along with SciPy and MiniSom for additional unsupervised learning analysis.

---

## 🎯 Objectives

The main objectives of this project are:

- Explore and understand the kidney disease dataset.
- Perform data preprocessing and quality checks.
- Handle missing values and categorical variables.
- Separate features and target variables.
- Apply feature scaling and preprocessing.
- Build and compare multiple classification algorithms.
- Evaluate classification performance using different metrics.
- Perform cross-validation and hyperparameter tuning.
- Apply clustering techniques to identify patient groups.
- Visualize clustering results.
- Explore dimensionality reduction using SVD.
- Apply Self-Organizing Maps (SOM) for unsupervised analysis.

---

## 📂 Project Structure

```text
Kidney-Disease-ML-Analysis/
│
├── Classification.ipynb
├── Clutsering.ipynb
├── kidney_dataset.csv
└── README.md
```

> **Note:** The clustering notebook is named `Clutsering.ipynb` in this project.

---

# 📊 Dataset

The project uses a kidney disease dataset stored in:

```text
kidney_dataset.csv
```

The dataset contains patient-related features and a target variable:

```text
CKD_Status
```

The target variable is used for the classification task.

The classification notebook first examines:

- Dataset dimensions
- Column names
- Data types
- Missing values
- Duplicate records
- Target class distribution

---

# 🤖 Part 1: Classification

## Objective

The classification task aims to predict the `CKD_Status` of patients using the available input features.

The following machine learning algorithms are implemented:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. Naive Bayes
7. Gradient Boosting

---

## 🔄 Classification Workflow

```text
Load Dataset
      ↓
Understand Dataset
      ↓
Check Missing Values
      ↓
Check Duplicate Records
      ↓
Analyze Target Variable
      ↓
Separate Features and Target
      ↓
Identify Numerical & Categorical Features
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Train Multiple Classification Models
      ↓
Generate Predictions
      ↓
Evaluate Models
      ↓
Compare Model Performance
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Evaluate Tuned Model
      ↓
ROC-AUC Analysis
```

---

## 🧹 Data Preprocessing

The classification workflow uses a preprocessing pipeline.

### Numerical Features

Numerical variables are standardized using:

```python
StandardScaler()
```

### Categorical Features

Categorical variables are converted into numerical representations using:

```python
OneHotEncoder(handle_unknown="ignore")
```

A `ColumnTransformer` is used to apply the appropriate preprocessing method to each feature type.

---

## ✂️ Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

The notebook uses:

```python
random_state=42
```

and:

```python
stratify=y
```

to maintain the class distribution between training and testing datasets.

---

# 🧠 Classification Models

## 1. Logistic Regression

Logistic Regression is used as a classification model to predict the CKD status.

The model is configured with:

```python
max_iter=1000
```

---

## 2. K-Nearest Neighbors

KNN classifies samples based on their nearest neighboring observations.

Configuration used:

```text
n_neighbors = 5
weights = uniform
metric = minkowski
```

---

## 3. Decision Tree

A Decision Tree is used to classify observations through a series of feature-based decision rules.

Configuration includes:

```text
criterion = gini
max_depth = 5
min_samples_split = 2
```

---

## 4. Random Forest

Random Forest combines multiple decision trees to perform classification.

The initial model uses:

```text
n_estimators = 100
criterion = gini
random_state = 42
```

---

## 5. Support Vector Machine

SVM is implemented using the RBF kernel.

Configuration:

```text
kernel = rbf
C = 1.0
gamma = scale
probability = True
```

---

## 6. Naive Bayes

Gaussian Naive Bayes is used for classification after preprocessing the feature matrix.

---

## 7. Gradient Boosting

Gradient Boosting is implemented using:

```text
n_estimators = 100
learning_rate = 0.1
max_depth = 3
```

---

# 📈 Model Evaluation

The classification models are compared using:

- Accuracy
- Precision
- Recall
- F1 Score

A comparison table is generated to evaluate the models.

The project also visualizes model accuracy using a bar plot.

---

## 🔲 Confusion Matrix

A confusion matrix is generated for the Random Forest model to visualize:

- Correct predictions
- Incorrect predictions
- Actual classes
- Predicted classes

---

# 🔁 Cross-Validation

The project applies **5-fold cross-validation** to evaluate model performance across multiple data splits.

The notebook calculates:

- Cross-validation scores
- Mean cross-validation accuracy
- Standard deviation

This provides an additional assessment of model performance beyond a single train-test split.

---

# ⚙️ Hyperparameter Tuning

`GridSearchCV` is used to tune the Random Forest classifier.

The following parameters are explored:

```text
n_estimators:
50, 100, 200

max_depth:
None, 5, 10

min_samples_split:
2, 5
```

The tuning process uses:

```text
5-fold cross-validation
```

and:

```text
accuracy
```

as the scoring metric.

The best parameter combination is then used to create the tuned model.

---

# 📉 ROC-AUC Analysis

The tuned model is further evaluated using:

- ROC curve
- ROC-AUC score

The model's predicted probability for the positive class is used to calculate ROC-AUC.

---

# 🔬 Part 2: Clustering

## Objective

The clustering analysis explores the kidney dataset using **unsupervised machine learning techniques**.

Unlike classification, clustering does not use `CKD_Status` as the target for model training.

The analysis attempts to identify natural groups and patterns within the patient feature data.

---

# 🔄 Clustering Workflow

```text
Load Dataset
      ↓
Check Missing Values
      ↓
Remove Target Variable
      ↓
Convert Categorical Features
      ↓
Handle Missing Values
      ↓
Standardize Features
      ↓
K-Means Clustering
      ↓
Cluster Visualization
      ↓
Elbow Method
      ↓
Hierarchical Clustering
      ↓
Dendrogram
      ↓
SVD
      ↓
SVD Visualization
      ↓
Self-Organizing Map (SOM)
      ↓
SOM Distance Map
```

---

# 🔵 K-Means Clustering

K-Means clustering is implemented with:

```text
n_clusters = 3
random_state = 42
n_init = 10
```

The resulting cluster labels are added to the original dataset as:

```text
KMeans_Cluster
```

The clusters are visualized using:

- Age
- GFR
- K-Means cluster assignment

---

# 📐 Elbow Method

The Elbow Method is used to investigate different numbers of clusters.

The notebook evaluates values of:

```text
K = 2 to 10
```

For each value of K, the model's inertia is calculated.

The resulting curve can be used to inspect how within-cluster variation changes with the number of clusters.

---

# 🌳 Hierarchical Clustering

Agglomerative Hierarchical Clustering is also applied.

The model uses:

```text
n_clusters = 3
```

The resulting cluster assignments are stored as:

```text
Hierarchical_Cluster
```

The clusters are visualized using:

- Age
- GFR
- Hierarchical cluster assignment

---

# 🌿 Dendrogram

A hierarchical clustering dendrogram is generated using SciPy.

The notebook uses:

```python
linkage(
    sample,
    method="ward"
)
```

The first 500 standardized observations are used for the dendrogram visualization.

The dendrogram provides a visual representation of the hierarchical relationships between observations.

---

# 📉 Singular Value Decomposition (SVD)

SVD is applied to the standardized feature matrix.

The notebook performs SVD using NumPy:

```python
U, S, VT = np.linalg.svd(X_scaled)
```

It also applies:

```python
TruncatedSVD(n_components=2)
```

to transform the feature data into two components.

The resulting two-dimensional representation is visualized using a scatter plot.

---

# 🗺️ Self-Organizing Maps (SOM)

A **Self-Organizing Map (SOM)** is implemented using the `MiniSom` library.

The SOM configuration includes:

```text
Grid size: 5 × 5
Sigma: 1.0
Learning rate: 0.5
Random seed: 42
Training iterations: 1000
```

After training, the winning SOM node for each observation is identified.

A SOM distance map is also generated to visualize the organization of the data across the SOM grid.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| SciPy | Hierarchical clustering |
| MiniSom | Self-Organizing Maps |
| Jupyter Notebook | Interactive analysis |

---

# 📦 Python Libraries

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy minisom jupyter
```

---

# ▶️ How to Run the Project

## Step 1: Clone the Repository

```bash
git clone https://github.com/karunasuryavansahi-commits/Kidney-Disease-ML-Analysis.git
```

## Step 2: Navigate to the Project Directory

```bash
cd Kidney-Disease-ML-Analysis
```

## Step 3: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy minisom jupyter
```

## Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

## Step 5: Run the Notebooks

Open:

```text
Classification.ipynb
```

for the supervised classification analysis.

Open:

```text
Clutsering.ipynb
```

for the unsupervised clustering analysis.

---

# 📊 Analysis Summary

### Classification

The classification notebook demonstrates a complete supervised machine learning workflow:

- Data exploration
- Data preprocessing
- Feature encoding
- Feature scaling
- Train-test splitting
- Multiple classification algorithms
- Model comparison
- Confusion matrix
- Cross-validation
- Hyperparameter tuning
- ROC-AUC analysis

### Clustering

The clustering notebook demonstrates several unsupervised learning approaches:

- K-Means clustering
- Elbow method
- Hierarchical clustering
- Dendrogram analysis
- Singular Value Decomposition
- Self-Organizing Maps

---

# 📌 Key Learning Outcomes

Through this project, the following machine learning concepts are demonstrated:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Categorical encoding
- Feature standardization
- Supervised learning
- Unsupervised learning
- Classification
- Clustering
- Model evaluation
- Cross-validation
- Hyperparameter optimization
- Dimensionality reduction
- Data visualization
- Machine learning pipelines

---

# ⚠️ Disclaimer

This project is intended for **educational and machine learning practice purposes**.

The models and analyses presented here should not be considered a clinical diagnostic system or used as a substitute for professional medical evaluation.

---

# 👩‍💻 Author

**Karuna Vijay Suryavanshi**

M.Sc. Bioinformatics  
Bharati Vidyapeeth (Deemed to be University), Pune

---

## ⭐ Project Focus

**Machine Learning | Healthcare Data Analysis | Classification | Clustering | Bioinformatics**

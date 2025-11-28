# Predicting Delivery Speed in a Food Delivery System Using Machine Learning Models

> **Group Project** - CSE422 Artificial Intelligence Lab (Spring 2025)  
> **Section:** 7 | **Group:** 8

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shoumikarc/Predicting-delivery-speed-in-a-food-delivery-system-using-machine-learning-models/blob/main/cse422_project.ipynb)

## 📊 Project Overview

This project focuses on predicting delivery speed in a food delivery system using machine learning models. By analyzing various factors such as the distance between the restaurant and the customer, weather conditions, traffic levels, and the courier's experience, we aim to accurately predict the delivery speed.

Through the application of essential data science techniques including data preprocessing, feature engineering, and model evaluation, this model provides valuable insights into the factors affecting delivery time and helps optimize food delivery operations.

## 🎯 Results

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| **Logistic Regression** | **80.7%** | 81.1% | 80.8% |
| Decision Tree | 69.8% | 70.0% | 69.8% |
| K-Nearest Neighbors | 68.3% | 68.9% | 68.3% |
| Neural Networks | 67.9% | 69.9% | 67.9% |

**Best Model:** Logistic Regression achieved the highest accuracy at 80.7%, demonstrating its effectiveness for this classification problem.

## 🚀 Running the Project

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge above to run directly in your browser - no setup required!

### Option 2: Local Jupyter Notebook
```bash
# Clone the repository
git clone https://github.com/shoumikarc/Predicting-delivery-speed-in-a-food-delivery-system-using-machine-learning-models.git
cd Predicting-delivery-speed-in-a-food-delivery-system-using-machine-learning-models

# Install dependencies
pip install -r requirements.txt

# Install Jupyter
pip install jupyter

# Run notebook
jupyter notebook cse422_project.ipynb
```

### Option 3: Convert to Python Script
```bash
# Install nbconvert
pip install nbconvert

# Convert notebook to script
jupyter nbconvert --to script cse422_project.ipynb

# Run the script
python cse422_project.py
```

## 📁 Project Structure
```
├── cse422_project.ipynb          # Main Jupyter notebook with complete analysis
├── CSE422 project report.pdf     # Detailed project report (11 pages)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 📊 Dataset

**Source:** Food Delivery Times Classification

**Features (8 predictor variables):**
- **Distance_km** - Distance between restaurant and customer (in km)
- **Preparation_Time_min** - Time taken to prepare the order (in minutes)
- **Courier_Experience_yrs** - Delivery person's experience (in years)
- **Weather** - Weather conditions (Clear, Rainy, Foggy, Windy)
- **Traffic_Level** - Traffic intensity (Low, Medium, High)
- **Time_of_Day** - Delivery time period (Morning, Afternoon, Evening, Night)
- **Vehicle_Type** - Type of delivery vehicle (Bike, Scooter, Car)

**Target Variable:**
- **Delivery_Speed** - Classification into three categories (Fast, Average, Slow)

**Dataset Size:** 1,000 delivery records

## 🛠️ Technologies & Libraries
```python
- Python 3.9+
- scikit-learn      # Machine learning models
- TensorFlow/Keras  # Neural network implementation
- pandas            # Data manipulation
- numpy             # Numerical computing
- matplotlib        # Visualization
- seaborn           # Statistical visualization
- Google Colab      # Development environment
```

## 📈 Methodology

### 1. Data Preprocessing
- **Missing Values:** Dropped rows with missing values (minimal impact due to dataset size)
- **Categorical Encoding:** LabelEncoder for categorical variables
- **Feature Scaling:** StandardScaler for normalization
- **Correlation Analysis:** Removed highly correlated features (threshold: 0.70)

### 2. Feature Engineering
- Correlation matrix heatmap analysis
- Identification of highly correlated feature pairs
- Feature selection to reduce multicollinearity
- StandardScaler application for optimal model performance

### 3. Dataset Splitting
- **Training Set:** 70%
- **Testing Set:** 30%
- **Random State:** 42 (for reproducibility)

### 4. Model Training & Evaluation
Implemented and compared four different models:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (k=5)
- Neural Networks (128 hidden units, 10 epochs)

### 5. Evaluation Metrics
- Accuracy Score
- Precision (weighted average)
- Recall (weighted average)
- Confusion Matrix
- ROC Curves with AUC scores
- Classification Reports

## 📊 Key Visualizations

The project includes comprehensive visualizations:
- ✅ **Correlation Heatmap** - Feature relationship analysis
- ✅ **Class Distribution** - Target variable frequency
- ✅ **Scatter Matrix** - Feature pair relationships
- ✅ **Model Accuracy Comparison** - Bar chart comparison
- ✅ **Precision vs Recall** - Model performance metrics
- ✅ **Confusion Matrices** - All four models
- ✅ **ROC Curves** - With AUC scores for each model

## 🔬 Model Comparison

### Observations
- **Logistic Regression** achieved the highest accuracy (80.7%), outperforming all other models
- **Decision Tree Classifier** showed good performance (69.8%) but is prone to overfitting
- **K-Nearest Neighbors** demonstrated sensitivity to feature scaling (68.3%)
- **Neural Networks** showed potential (67.9%) but require additional tuning

### Key Findings
- Simpler models (Logistic Regression) performed competitively with more complex models
- Feature scaling significantly improved model performance
- Data preprocessing plays a crucial role in model accuracy
- Class imbalance was addressed through proper evaluation metrics

## 👥 Team Members (Group 8)

| Name | Student ID | GitHub |
|------|------------|--------|
| Md. Shariar Islam Shuvo | 21201105 | [@shariar-username](#) |
| Shakib Shadman Shoumik | 22101057 | [@shoumikarc](https://github.com/shoumikarc) |
| Md Nurullah | 21201262 | [@nurullah-username](#) |

## 📄 Full Report

For detailed analysis, methodology, experimental results, and conclusions, please refer to the comprehensive [Project Report](CSE422%20project%20report.pdf).

**Report Highlights:**
- Introduction and problem statement
- Dataset description and analysis
- Preprocessing techniques
- Model implementation details
- Comparative analysis
- Conclusions and key takeaways

## 🎓 Academic Context

**Course:** CSE422 - Artificial Intelligence  
**Lab Section:** 7  
**Group:** 8  
**Semester:** Spring 2025  
**Institution:** BRAC University  
**Instructors:** Farhan Faruk, Md. Mahadi Hasan  
**Submission Date:** May 6, 2025

## 💡 Key Takeaways

- ✅ Data preprocessing is crucial for enhancing model performance
- ✅ Feature engineering significantly improves predictive power
- ✅ Model selection depends on dataset characteristics and problem requirements
- ✅ Simpler models often provide competitive results with proper feature engineering
- ✅ Logistic Regression's strong performance demonstrates its suitability for linear relationships

## 🎯 Future Improvements

- Implementation of ensemble methods (Random Forest, XGBoost)
- Hyperparameter tuning using GridSearchCV
- Handling class imbalance with SMOTE
- Feature importance analysis
- Real-time prediction system deployment

## 🤝 Collaboration Note

This is a collaborative academic project completed as part of CSE422 coursework at BRAC University. Each team member maintains a copy of this repository for portfolio purposes. All team members contributed equally to the project.

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue in this repository
- Connect via GitHub profile

---

⭐ **If you found this project helpful, please consider giving it a star!**

## 📜 License

This project is submitted as academic coursework for CSE422 - Artificial Intelligence at BRAC University.

---

**Project Repository:** https://github.com/shoumikarc/Predicting-delivery-speed-in-a-food-delivery-system-using-machine-learning-models

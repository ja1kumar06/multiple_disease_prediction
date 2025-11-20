# Multiple Disease Prediction Using Machine Learning

### Welcome to the Multiple Disease Prediction System — A comprehensive machine learning project that predicts the risk of multiple common diseases from patient data with high accuracy.

---

## Supported Diseases

Our system currently predicts the following diseases with associated model accuracy:

| Disease                  | Accuracy   |
|--------------------------|------------|
| Heart Disease            | ~96.1%     |
| Diabetes                 | ~89.2%     |
| Parkinson's Disease      | 100%       |
| Chronic Kidney Disease   | 100%       |
| Combined                 | 96.3%      |

---

## Technologies Used

- Python 3.x
- Streamlit (Interactive Web Interface)
- Scikit-Learn (Machine Learning Models)
- Pandas & NumPy (Data Processing)
- Flask (Optional API Backend)
- Random Forest Classifiers

---

## Features

- Interactive user-friendly web app to enter patient data and get real-time predictions.
- Synthetic but clinically relevant datasets for each disease.
- Models trained to deliver ultra-high accuracy.
- Extensible modular codebase for adding new diseases or data sources.
- Pickled models and accuracy snapshots for fast deployment.

---

## Installation

### Step 1: Clone the repository

git clone https://github.com/ja1kumar06/multiple_disease_prediction_using_ML.git

cd multiple_disease_prediction_using_ML


### Step 2: Install required packages

pip install -r requirements.txt


---

## Usage

Launch the interactive Streamlit web application:

streamlit run app.py


- Select a disease from the sidebar.
- Enter patient clinical parameters.
- Click the predict button for risk assessment.
- View risk probability and interpretative insights.

---

## 📁 Project Structure

- [`app.py`](./app.py) → Streamlit web app frontend  
- [`multiple_disease_prediction_model.py`](./multiple_disease_prediction_model.py) → Model training scripts  
- [`models.pkl`](./models.pkl) → Serialized trained models  
- [`accuracies.pkl`](./accuracies.pkl) → Model accuracy data  
- [`datasets/`](./datasets) → Folder containing synthetic datasets  
  - [`heart_disease.csv`](./datasets/heart_disease.csv) → Heart disease synthetic dataset  
  - [`diabetes.csv`](./datasets/diabetes.csv) → Diabetes synthetic dataset  
  - [`parkinsons.csv`](./datasets/parkinsons.csv) → Parkinson’s disease synthetic dataset  
  - [`ckd.csv`](./datasets/ckd.csv) → CKD synthetic dataset  
- [`requirements.txt`](./requirements.txt) → Python package dependencies  
- [`README.md`](./README.md) → Project description  
- [`.gitattributes`](./.gitattributes) → Git settings for consistent line endings  
- [`.gitignore`](./.gitignore) → Files/folders ignored by Git  

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

- Fork the repo
- Create a new branch for your feature/fix
- Submit a pull request describing your changes



---

## Contact

For questions, suggestions or collaboration, reach out via:  
**Email:** jaiswaroop1259@gmail.com

---

Thank you for visiting and supporting this project! ⭐




Multiple Disease Prediction Using Machine Learning
Welcome to the Multiple Disease Prediction System — a comprehensive machine learning project that predicts the risk of multiple common diseases from patient data with high accuracy.

Supported Diseases
Disease	Accuracy
Heart Disease	~96.1%
Diabetes	~89.2%
Parkinson's Disease	100%
Chronic Kidney Disease	100%
Combined	96.3%

Technologies Used

Python 3.x

Streamlit (Interactive Web Interface)

Scikit-Learn (Machine Learning Models)

Pandas & NumPy (Data Processing)

Flask (Optional API Backend)

Random Forest Classifiers

Docker (Containerization & portability)

AWS S3 (Cloud storage for models/data)

AWS CLI (Automated asset retrieval)

Features

Interactive, user-friendly web app for real-time disease risk prediction.

Synthetic but clinically relevant datasets for each disease.

Highly accurate models, fast deployment with pickled files.

Modular codebase for easy extension to more diseases or sources.

Cloud integration using Docker and AWS S3, for scalable production.

Installation

Step 1: Clone the repository

git clone https://github.com/ja1kumar06/multiple_disease_prediction_using_ML.git

cd multiple_disease_prediction_using_ML

Step 2: Install required packages

pip install -r Requirements.txt

Usage — Local (Streamlit)

streamlit run app.py

Select a disease from the sidebar.

Enter patient clinical parameters.

Click "Predict" for risk assessment.

View probability and interpretative insights.

Usage — Docker & AWS S3 Integration

This project supports running in Docker with automatic downloading of models and datasets from AWS S3.

Prerequisites:

AWS S3 bucket with required files (models.pkl, accuracies.pkl, all datasets).

AWS credentials configured (IAM role, environment variables, or aws configure).

Build and Run Docker Container:

#bash

docker build -t disease-prediction-app .

docker run -e S3_BUCKET=your-s3-bucket-name disease-prediction-app

What happens:

Docker container downloads all required assets from S3 into expected locations (via download_s3_assets.sh).

Your app starts with the latest models and datasets from the cloud.

📁 Project Structure

Multiple_Disease_Prediction/
├── app.py
├── multiple_disease_prediction_model.py
├── Requirements.txt
├── README.md
├── Dockerfile                # <---- NEW: Docker deployment
├── download_s3_assets.sh     # <---- NEW: S3 asset sync script
├── datasets/
│   ├── heart.csv
│   ├── diabetes.csv
│   ├── parkinsons.csv
│   └── ckd.csv
├── models.pkl
├── accuracies.pkl
├── .gitattributes
├── .gitignore

Contributing

Contributions, bug reports, and feature requests are welcome!

Fork the repo

Create a feature branch

Submit your pull request with a clear description

Please follow coding standards and document your changes.

Contact & Support

For questions or collaboration, reach out via:

Email: jaiswaroop1259@gmail.com

Thank you for visiting and supporting this project! ⭐
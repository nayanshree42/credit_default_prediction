# Credit Default Prediction 

A machine learning model that predicts the likelihood of credit default using customer financial data.

## Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/your-username/credit-default-prediction.git
cd credit-default-prediction
pip install -r requirements.txt

## Usage
Run the FastAPI server:
```bash
python -m uvicorn app.main:app --reload

## Features
- Predicts credit default risk
- REST API with FastAPI
- Interactive Swagger UI

## Project Structure
credit_default_prediction/
├── data/                # dataset will go here
├── notebooks/           # optional for EDA
├── src/                 # all code files
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── predict.py
├── app/                 # deployment code
│   ├── main.py
├── requirements.txt
├── README.md## Contributing
Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

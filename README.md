## Brief info

This app is a Machine Learning model application which is used to predict the credit score of an individual based on some input features. A credit score is a numerical expression based on a level analysis of a person's credit files, to represent the creditworthiness of an individual. Basically, credit score is a prediction of your credit behavior, such as how likely you are to pay a loan back on time, based on information from your credit reports. 

Feature selection, EDA and feature engineering can be viewed from jupyter notebooks. Model used for deployment is XgBoost model due to its memory friendly and light weight nature. This model takes 16 input features and based on which predicts a single output of credit score of a person.

Live link of website is available here => <a target="_blank" href="https://somil-credit-score-classification.streamlit.app/">live link</a>.

## Running Locally

If you want to run the app locally you can clone the project

```bash
  git clone https://github.com/Somil7Jain/Credit-Score-Classification
```

Install the requirements

```bash
  pip install -r requirements.txt
```

Enter the project directory

```bash
  cd credit-score-classification
```

Run the Streamlit App

```bash
  streamlit run app/main.py
```

## Project Organization

```txt
├── README.md          <- README of this project.
│
├── app                <- main code for application.
│
├── dataset            <- dataset on which models are trained (csv file).
│
├── files              <- containing some meta data csv files.
│
├── icons              <- icon for website created using streamlit.
│
├── info               <- two pdfs containing information regarding project, corresponding to two jupyter notebooks.
│
├── models             <- contains final model used for deployment.
│
├── notebooks          <- Jupyter notebooks containing training code.
│
└── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                          generated with `pip freeze > requirements.txt`
```

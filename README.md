# House Price Prediction — End-to-End ML Pipeline

A production-ready machine learning pipeline that predicts California house prices, served via a REST API and containerised with Docker.

## Tech Stack
- Python
- scikit-learn
- pandas / numpy
- FastAPI
- Docker
- joblib
- MLflow

## Pipeline Steps
1. Load and explore data
2. Clean and scale features
3. Train model
4. Evaluate (MAE: $53k, R2: 0.57)
5. Save model to disk
6. Serve predictions via FastAPI
7. Containerise with Docker

## Quick Start

Run locally:
pip install -r requirements.txt
uvicorn src.api:app --reload

Run with Docker:
docker build -t house-price-api:latest .
docker run -p 8000:8000 house-price-api:latest

## Author
Soorya Prabhu - MSc Artificial Intelligence, Brunel University London

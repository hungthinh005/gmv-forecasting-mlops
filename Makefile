# Makefile for GMV Forecasting MLOps

.PHONY: help install setup clean test lint format train evaluate api docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make setup         - Setup project directories"
	@echo "  make clean         - Clean generated files"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make prepare-data  - Prepare data"
	@echo "  make train         - Train models"
	@echo "  make evaluate      - Evaluate models"
	@echo "  make api           - Run API server"
	@echo "  make mlflow        - Run MLflow UI"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker containers"

install:
	pip install -r requirements.txt

setup:
	mkdir -p data/raw data/processed data/predictions
	mkdir -p models logs outputs mlruns
	mkdir -p outputs/plots
	@echo "Project directories created"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "Cleaned generated files"

test:
	pytest tests/ -v --cov=src --cov-report=html

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

prepare-data:
	python src/data/prepare_data.py --config config/config.yaml

train:
	python src/models/train.py --config config/config.yaml

evaluate:
	python src/evaluation/evaluate.py --config config/config.yaml

api:
	uvicorn src.api.main:app --reload --port 8000

mlflow:
	mlflow ui --backend-store-uri ./mlruns

docker-build:
	docker-compose -f deployment/docker-compose.yml build

docker-run:
	docker-compose -f deployment/docker-compose.yml up

docker-down:
	docker-compose -f deployment/docker-compose.yml down

docker-logs:
	docker-compose -f deployment/docker-compose.yml logs -f api

# Development workflow
dev-setup: install setup
	@echo "Development environment ready!"

dev-run: prepare-data train evaluate api

# Production workflow
prod-build: clean docker-build

prod-deploy: docker-run

# CI/CD
ci: lint test
	@echo "CI checks passed!"


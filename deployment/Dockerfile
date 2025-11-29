# Single-stage Dockerfile for GMV Forecasting API with Prophet support

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including Prophet/CmdStan requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq5 \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Initialize Prophet's CmdStan backend (critical for Prophet to work)
RUN python -c "from cmdstanpy import install_cmdstan; install_cmdstan(overwrite=True, verbose=True)"

# Verify Prophet works
RUN python -c "from prophet import Prophet; import pandas as pd; df = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=10), 'y': range(10)}); m = Prophet(); m.fit(df); print('✓ Prophet initialized successfully!')"

# Copy application code
COPY config/ config/
COPY src/ src/
COPY models/ models/

# Create necessary directories
RUN mkdir -p logs data/raw data/processed data/predictions outputs mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Tell CmdStanPy where to find CmdStan installation
ENV CMDSTAN=/root/.cmdstan

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

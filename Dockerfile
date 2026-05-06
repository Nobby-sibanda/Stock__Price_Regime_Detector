FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy repo into a subdirectory named stock_regime_detector
# so Python can import it as a proper package
COPY . stock_regime_detector/

RUN pip install --no-cache-dir \
    -r stock_regime_detector/requirements.txt \
    flask

# /workspace is on PYTHONPATH so 'import stock_regime_detector' resolves correctly
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

EXPOSE 5000
WORKDIR /workspace/stock_regime_detector
CMD ["python", "app.py"]

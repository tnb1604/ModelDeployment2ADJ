# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and the pre-trained model
COPY app.py .
COPY model/ model/

# Expose the port Flask/Gunicorn will listen on
EXPOSE 5000

# Run with Gunicorn (production WSGI server, not Flask dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]

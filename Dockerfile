# =====================================
#   AI Doctor Flask App (Python 3.11.0)
# =====================================
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files into container
COPY . .

# Upgrade build tools and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Expose Flask default port
EXPOSE 5000

# Run Flask app (Render automatically sets PORT)
CMD ["python", "app.py"]

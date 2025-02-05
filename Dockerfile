# Use a full Python base image to avoid extra dependencies
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy the requirements first to leverage Docker's build cache
COPY requirements.txt /app/

# Install necessary system dependencies and Python packages
RUN apt-get update && apt-get install -y gcc && \
    pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir numpy==1.24.4 scipy scikit-learn surprise && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up apt and cache to reduce image size
    apt-get remove -y gcc && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . /app

# Set environment variables
ENV FLASK_APP=Flask.py  
ENV FLASK_RUN_HOST=0.0.0.0

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["flask", "run"]

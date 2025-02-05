# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV FLASK_APP=Flask.py  
ENV FLASK_RUN_HOST=0.0.0.0

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["flask", "run"]

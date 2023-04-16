# Use an official NVIDIA CUDA runtime as a parent image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Define environment variable
ENV PYTHONUNBUFFERED 1

# Run the script when the container launches
CMD ["python3", "scripts/AKTrain.py"]

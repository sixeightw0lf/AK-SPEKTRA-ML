# Use an official PyTorch image as the parent image
FROM pytorch/pytorch:1.10.0-cuda11.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run tuner_script.py when the container launches
CMD ["python", "AKTuner_4Q.py"]

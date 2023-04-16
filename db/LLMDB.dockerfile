# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install Faiss and Faiss-gpu for efficient similarity search
RUN pip install faiss faiss-gpu

# Install ChromaDB and its dependencies
RUN pip install chromadb

# Install uvicorn for running the ChromaDB server
RUN pip install uvicorn

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables
ENV CHROMA_DB_IMPL=clickhouse
ENV CLICKHOUSE_HOST=clickhouse
ENV CLICKHOUSE_PORT=8123

# Run the ChromaDB server when the container launches
CMD ["uvicorn", "chromadb.app:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "8000", "--log-config", "log_config.yml"]

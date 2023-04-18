#!/bin/bash

# Update package lists and install required dependencies
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker GPG key and repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Add NVIDIA GPG key and repository
curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -sL https://nvidia.github.io/nvidia-docker/$(. /etc/os-release;echo $ID$VERSION_ID)/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update package lists again
sudo apt-get update

# Install Docker
sudo apt-get install -y docker-ce

# Install NVIDIA Docker
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker

# Enable Docker daemon on boot
sudo systemctl enable docker

# Create docker-compose.yml file
cat <<EOT > docker-compose.yml
version: "2.4"

services:
  model_trainer:
    image: ak-spektra-r1
    ports:
      - "5000:5000"
    volumes:
      - akdata:/app/akdata
    env_file:
      - .env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  akdata:
EOT

# Run Docker Compose
sudo docker-compose up -d

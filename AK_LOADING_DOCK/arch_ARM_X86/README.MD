# Arch Linux Installation Scripts

This repository contains two installation scripts to set up an Arch Linux system with either a KDE desktop environment or a Docker server.

## Prerequisites

Before proceeding, make sure you have installed a minimal Arch Linux base system. You can follow the [Arch Linux Installation Guide](https://wiki.archlinux.org/title/Installation_guide) for instructions.

# Usage

## Docker:

### Dockerfile for Arch Linux base

Create a file named `Dockerfile_base` in the root directory of your project with the following contents:

.dockerfile bash

`FROM archlinux:base  RUN pacman -Syu --noconfirm \   && pacman -S --noconfirm base-devel git \   && pacman -Scc --noconfirm`

### Dockerfile for Arch Linux KDE Desktop

Create a file named `Dockerfile_kde` in the root directory of your project with the following contents:

.dockerfile bash

`FROM archlinux_base  COPY install_kde_desktop.sh /install_kde_desktop.sh  RUN chmod +x /install_kde_desktop.sh \   && /install_kde_desktop.sh`

### Dockerfile for Arch Linux Docker Server

Create a file named `Dockerfile_docker` in the root directory of your project with the following contents:

.dockerfile bash

`FROM archlinux_base  COPY install_docker_server.sh /install_docker_server.sh  RUN chmod +x /install_docker_server.sh \   && /install_docker_server.sh`

### Docker Compose

Create a `docker-compose.yml` file in the root directory of your project with the following contents:

yaml

`version: '3'  services:   kde_desktop:     build:       context: .       dockerfile: Dockerfile_kde     image: archlinux_kde_desktop    docker_server:     build:       context: .       dockerfile: Dockerfile_docker     image: archlinux_docker_server`

### Python Automation Script

Create a file named `deploy.py` in the root directory of your project with the following contents:

python

`import os import sys  def build_images():     os.system("docker build -t archlinux_base -f Dockerfile_base .")     os.system("docker-compose build")  def deploy_locally():     os.system("docker-compose up -d")  def main():     if len(sys.argv) != 3:         print("Usage: python deploy.py [local|cloud] <path_to_project_root>")         sys.exit(1)      deployment_type = sys.argv[1]     project_root = sys.argv[2]      os.chdir(project_root)          build_images()      if deployment_type == "local":         deploy_locally()     elif deployment_type == "cloud":         print("Cloud deployment not implemented.")     else:         print("Invalid deployment type. Use 'local' or 'cloud'.")  if __name__ == "__main__":     main()`

This script automates the process of building Docker images and deploying them locally. You can modify the script to deploy to the cloud based on your preferred cloud provider.

### To run the script, execute the following command:

bash

`python deploy.py local /path/to/your/project/root`

### 1. Arch Linux KDE Desktop

To set up an Arch Linux system with the KDE desktop environment, follow these steps:

1. Clone the repository or download the `install_kde_desktop.sh` script.

2. Open a terminal, navigate to the directory containing the script, and make it executable:

```chmod +x install_kde_desktop.sh```

3. Run the script:

```./install_kde_desktop.sh```

4. Follow the on-screen prompts to create a user and set a password.

5. Reboot your system when the script finishes executing.

### 2. Arch Linux Docker Server

To set up an Arch Linux system with a Docker server, follow these steps:

1. Clone the repository or download the `install_docker_server.sh` script.

2. Open a terminal, navigate to the directory containing the script, and make it executable:

chmod +x install_docker_server.sh

3. Run the script:

./install_docker_server.sh


4. Follow the on-screen prompts to create a user and set a password.

5. Reboot your system when the script finishes executing.

## Customizations

These scripts provide a minimal setup for an Arch Linux KDE desktop and Docker server. You can customize and extend the installation process by modifying the scripts as needed. For more information, refer to the [Arch Linux Wiki](https://wiki.archlinux.org/).

## License

This project is licensed under the MIT License.

To create Dockerfiles and Docker Compose files for each script, you'll first need to create a Dockerfile for the base Arch Linux installation. Then, you can create Dockerfiles for the KDE Desktop and Docker Server installations, which will build on top of the base Arch Linux image. Finally, you'll create a Python script to automate the process.



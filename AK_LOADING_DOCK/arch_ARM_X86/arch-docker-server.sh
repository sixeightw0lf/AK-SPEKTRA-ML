#!/bin/bash
set -e

# Update the system
pacman -Syu --noconfirm

# Install essential packages
pacman -S --noconfirm base-devel git

# Install Docker
pacman -S --noconfirm docker

# Enable and start Docker service
systemctl enable docker.service
systemctl start docker.service

# Add a user to the Docker group
read -p "Enter your desired username: " username
useradd -m -G wheel,users,docker -s /bin/bash "$username"
passwd "$username"

# Configure sudoers file
echo '%wheel ALL=(ALL) ALL' | EDITOR='tee -a' visudo

# Reboot the system
reboot

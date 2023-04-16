#!/bin/bash
set -e

# Update the system
pacman -Syu --noconfirm

# Install essential packages
pacman -S --noconfirm base-devel git

# Install Xorg and graphics drivers
pacman -S --noconfirm xorg-server xorg-xinit xorg-apps xf86-video-vesa

# Install KDE Plasma and essential applications
pacman -S --noconfirm plasma-desktop dolphin konsole kate kde-system-meta kde-graphics-meta kde-multimedia-meta kde-network-meta kde-pim-meta kde-sdk-meta

# Enable SDDM display manager
systemctl enable sddm.service

# Create a user and set a password
read -p "Enter your desired username: " username
useradd -m -G wheel,users -s /bin/bash "$username"
passwd "$username"

# Configure sudoers file
echo '%wheel ALL=(ALL) ALL' | EDITOR='tee -a' visudo

# Reboot the system
reboot

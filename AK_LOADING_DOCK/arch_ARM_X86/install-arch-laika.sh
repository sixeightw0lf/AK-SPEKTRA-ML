#!/bin/bash
set -e

# Update the system
pacman -Syu --noconfirm

# Install essential packages
pacman -S --noconfirm base-devel git

# Install Nginx
pacman -S --noconfirm nginx

# Install Node.js
pacman -S --noconfirm nodejs npm

# Install Python
pacman -S --noconfirm python python-pip

# Configure Nginx for better security and reverse proxy
sed -i 's/# server_tokens off;/server_tokens off;/' /etc/nginx/nginx.conf
echo 'add_header X-Content-Type-Options nosniff;' >> /etc/nginx/conf.d/security.conf
echo 'add_header X-Frame-Options "SAMEORIGIN";' >> /etc/nginx/conf.d/security.conf
echo 'add_header X-XSS-Protection "1; mode=block";' >> /etc/nginx/conf.d/security.conf
echo 'add_header X-Robots-Tag none;' >> /etc/nginx/conf.d/security.conf
echo 'add_header X-Download-Options noopen;' >> /etc/nginx/conf.d/security.conf
echo 'add_header X-Permitted-Cross-Domain-Policies none;' >> /etc/nginx/conf.d/security.conf
echo 'add_header Referrer-Policy "no-referrer-when-downgrade";' >> /etc/nginx/conf.d/security.conf

# Configure Nginx as a reverse proxy for Node.js and Python applications
echo 'upstream nodejs {' >> /etc/nginx/conf.d/upstream.conf
echo '    server 127.0.0.1:3000;' >> /etc/nginx/conf.d/upstream.conf
echo '}' >> /etc/nginx/conf.d/upstream.conf

echo 'upstream python {' >> /etc/nginx/conf.d/upstream.conf
echo '    server 127.0.0.1:8000;' >> /etc/nginx/conf.d/upstream.conf
echo '}' >> /etc/nginx/conf.d/upstream.conf

# Enable and start Nginx
systemctl enable nginx
systemctl start nginx

# Install and configure firewalld
pacman -S --noconfirm firewalld
systemctl enable firewalld
systemctl start firewalld

# Open necessary ports
firewall-cmd --add-service=http --permanent
firewall-cmd --add-service=https --permanent
firewall-cmd --reload

# Create a user and set a password
read -p "Enter your desired username: " username
useradd -m -G wheel,users -s /bin/bash "$username"
passwd "$username"

# Configure sudoers file
echo '%wheel ALL=(ALL) ALL' | EDITOR='tee -a' visudo

# Reboot the system
reboot

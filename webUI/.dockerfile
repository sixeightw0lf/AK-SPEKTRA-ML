# Use an official Node.js runtime as a parent image
FROM node:14

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json /app/

# Install any needed packages
RUN npm install

# Copy the rest of the application code
COPY . /app

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Define environment variable
ENV NAME WebUI

# Run the command to start the web user interface
CMD ["npm", "start"]

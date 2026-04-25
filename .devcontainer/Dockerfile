# Use the official Python image from the Docker Hub
FROM mcr.microsoft.com/devcontainers/python:3.11

# Set the working directory
WORKDIR /workspaces

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Default command
CMD ["sleep", "infinity"]
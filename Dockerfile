# Use a base image with Python
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Set the command to run your application (modify if needed later)
CMD ["python", "src/app/app.py"]
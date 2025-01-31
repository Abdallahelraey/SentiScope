# Use official lightweight Python image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app  

# Copy only the requirements first (to leverage Docker caching)
COPY requirements_prod.txt .  

# Install dependencies and MLflow explicitly
RUN pip install --no-cache-dir -r requirements_prod.txt \
    && pip install --no-cache-dir mlflow

# Copy all project files into the container
COPY . .  

# Fix potential Windows line endings in start.sh
RUN sed -i 's/\r$//' /app/start.sh

# Make the script executable
RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 5000 8080

# Run the script
CMD ["/app/start.sh"]
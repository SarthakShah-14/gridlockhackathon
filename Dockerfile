FROM python:3.10-slim

# Install system dependencies (like unzip)
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy current directory contents
COPY . /code

# Give full permissions to the code folder for dynamic file writing (PDF, logs) on Hugging Face
RUN chmod -R 777 /code

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory and unzip model parts
RUN mkdir -p models && \
    unzip -o models_part1.zip -d models/ && \
    unzip -o models_part2.zip -d models/ && \
    unzip -o models_part3.zip -d models/

# Modify port and bind to 0.0.0.0 in dashboard/app.py for Hugging Face Spaces compatibility
RUN python -c "with open('dashboard/app.py', 'r') as f: c = f.read(); c = c.replace('PORT = 8085', 'PORT = 7860').replace('(\"\", port)', '(\"0.0.0.0\", port)'); \
    with open('dashboard/app.py', 'w') as f: f.write(c)"

# Expose the port Hugging Face expects
EXPOSE 7860

# Run the app
CMD ["python", "dashboard/app.py"]

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y unzip libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY . /code
RUN chmod -R 777 /code

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models && \
    unzip -o models_part1.zip -d models/ && \
    unzip -o models_part2.zip -d models/ && \
    unzip -o models_part3.zip -d models/

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "dashboard/app.py"]

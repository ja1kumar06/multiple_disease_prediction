FROM python:3.11

WORKDIR /app

COPY Requirements.txt .
RUN pip install -r Requirements.txt

COPY . .

RUN pip install awscli
RUN chmod +x /app/download_s3_assets.sh

ENV S3_BUCKET=your-bucket-name

ENTRYPOINT ["/bin/bash", "-c", "bash /app/download_s3_assets.sh && streamlit app.py"]

#!/bin/bash

BUCKET="${S3_BUCKET:-your-bucket-name}"

aws s3 cp s3://${BUCKET}/models.pkl ./models.pkl
aws s3 cp s3://${BUCKET}/accuracies.pkl ./accuracies.pkl

mkdir -p datasets

aws s3 cp s3://${BUCKET}/heart.csv ./datasets/heart.csv
aws s3 cp s3://${BUCKET}/diabetes.csv ./datasets/diabetes.csv
aws s3 cp s3://${BUCKET}/parkinsons.csv ./datasets/parkinsons.csv
aws s3 cp s3://${BUCKET}/ckd.csv ./datasets/ckd.csv

echo "Downloaded S3 assets."

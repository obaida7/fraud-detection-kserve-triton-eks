output "bucket_name" {
  value = aws_s3_bucket.model_bucket.id
}

output "bucket_arn" {
  value = aws_s3_bucket.model_bucket.arn
}

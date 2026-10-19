aws_region   = "us-east-2"
cluster_name = "fraud-detection-eks"
bucket_name  = "fraud-detection-triton-227246780492"
environment  = "production"

tags = {
  Project     = "fraud-detection-kserve-triton-eks"
  Environment = "production"
  ManagedBy   = "Terraform"
  Engine      = "Triton-KServe-Inference"
}

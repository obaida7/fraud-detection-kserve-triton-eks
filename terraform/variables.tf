variable "aws_region" {
  type        = string
  default     = "us-east-2"
  description = "Selected AWS Region for deployment"
}

variable "cluster_name" {
  type        = string
  default     = "fraud-detection-eks"
  description = "EKS Cluster Name for Triton & KServe workloads"
}

variable "bucket_name" {
  type        = string
  default     = "fraud-detection-triton-227246780492"
  description = "S3 bucket name for Triton model repository and runtime telemetry"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment"
}

variable "tags" {
  type = map(string)
  default = {
    Project     = "fraud-detection-kserve-triton-eks"
    Environment = "production"
    ManagedBy   = "Terraform"
    Workload    = "Triton-KServe-Serving"
  }
  description = "Common resource tags"
}

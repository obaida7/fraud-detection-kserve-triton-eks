output "aws_region" {
  value       = var.aws_region
  description = "Selected AWS Region"
}

output "cluster_name" {
  value       = module.eks.cluster_name
  description = "EKS Cluster Name"
}

output "cluster_endpoint" {
  value       = module.eks.cluster_endpoint
  description = "EKS API Server Endpoint"
}

output "s3_model_bucket" {
  value       = module.storage.bucket_name
  description = "S3 Bucket hosting Triton Model Repository"
}

output "kubeconfig_command" {
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${module.eks.cluster_name}"
  description = "Command to configure kubectl context"
}

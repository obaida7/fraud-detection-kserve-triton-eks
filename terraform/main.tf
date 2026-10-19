terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ════════════════════════════════════════════════════════════════════════
# 1. STORAGE MODULE (S3 Triton Model Repository & Data Capture)
# ════════════════════════════════════════════════════════════════════════
module "storage" {
  source      = "./modules/storage"
  bucket_name = var.bucket_name
  tags        = var.tags
}

# ════════════════════════════════════════════════════════════════════════
# 2. VPC MODULE (Public & Private Subnets for EKS & ALB Ingress)
# ════════════════════════════════════════════════════════════════════════
module "vpc" {
  source       = "./modules/vpc"
  name_prefix  = var.cluster_name
  cluster_name = var.cluster_name
  tags         = var.tags
}

# ════════════════════════════════════════════════════════════════════════
# 3. IAM MODULE (EKS Cluster Role, Worker Node Role, IRSA Policies)
# ════════════════════════════════════════════════════════════════════════
module "iam" {
  source       = "./modules/iam"
  cluster_name = var.cluster_name
  tags         = var.tags
}

# ════════════════════════════════════════════════════════════════════════
# 4. EKS CLUSTER MODULE (Kubernetes 1.30 Control Plane & Node Group)
# ════════════════════════════════════════════════════════════════════════
module "eks" {
  source           = "./modules/eks"
  cluster_name     = var.cluster_name
  cluster_role_arn = module.iam.cluster_role_arn
  node_role_arn    = module.iam.node_role_arn
  subnet_ids       = module.vpc.private_subnet_ids
  tags             = var.tags

  depends_on = [
    module.vpc,
    module.iam
  ]
}

# ════════════════════════════════════════════════════════════════════════
# 5. UPLOAD TRITON MODEL ARTIFACTS TO S3 MODEL REPOSITORY
# ════════════════════════════════════════════════════════════════════════
resource "aws_s3_object" "triton_config" {
  bucket = module.storage.bucket_name
  key    = "triton-models/fraud_detection/config.pbtxt"
  source = "${path.module}/../model_repository/fraud_detection/config.pbtxt"
  etag   = filemd5("${path.module}/../model_repository/fraud_detection/config.pbtxt")

  depends_on = [module.storage]
}

resource "aws_s3_object" "triton_model_onnx" {
  bucket = module.storage.bucket_name
  key    = "triton-models/fraud_detection/1/model.onnx"
  source = "${path.module}/../model_repository/fraud_detection/1/model.onnx"
  etag   = filemd5("${path.module}/../model_repository/fraud_detection/1/model.onnx")

  depends_on = [module.storage]
}

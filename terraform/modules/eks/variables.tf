variable "cluster_name" {
  type        = string
  description = "EKS Cluster Name"
}

variable "cluster_version" {
  type        = string
  default     = "1.30"
  description = "Kubernetes Version"
}

variable "cluster_role_arn" {
  type        = string
  description = "IAM Role ARN for EKS Cluster Control Plane"
}

variable "node_role_arn" {
  type        = string
  description = "IAM Role ARN for Managed Worker Nodes"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for EKS Cluster and Worker Nodes"
}

variable "instance_types" {
  type        = list(string)
  default     = ["t3.medium"]
  description = "EC2 Instance Types for EKS worker nodes"
}

variable "desired_size" {
  type    = number
  default = 2
}

variable "min_size" {
  type    = number
  default = 1
}

variable "max_size" {
  type    = number
  default = 5
}

variable "tags" {
  type    = map(string)
  default = {}
}

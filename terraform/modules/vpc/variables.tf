variable "name_prefix" {
  type        = string
  description = "Prefix for VPC resource names"
}

variable "cidr_block" {
  type        = string
  default     = "10.0.0.0/16"
  description = "VPC CIDR block"
}

variable "cluster_name" {
  type        = string
  description = "EKS Cluster Name for ELB discovery tags"
}

variable "tags" {
  type    = map(string)
  default = {}
}

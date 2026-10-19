variable "bucket_name" {
  type        = string
  description = "Name of S3 bucket for model artifacts and transaction captures"
}

variable "tags" {
  type    = map(string)
  default = {}
}

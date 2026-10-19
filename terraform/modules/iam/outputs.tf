output "cluster_role_arn" {
  value = aws_iam_role.cluster_role.arn
}

output "node_role_arn" {
  value = aws_iam_role.node_role.arn
}

output "model_s3_policy_arn" {
  value = aws_iam_policy.model_s3_read.arn
}

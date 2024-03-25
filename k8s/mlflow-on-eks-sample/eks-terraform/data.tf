# from https://github.com/NVIDIA/nvidia-terraform-modules/blob/main/eks/data.tf and modified to fit the project

data "aws_availability_zones" "available" {}
data "aws_region" "current" {}

data "aws_ami" "lookup" {
  most_recent = true
  owners      = local.ami_lookup.owners
  dynamic "filter" {
    for_each = local.ami_lookup.filters
    content {
      name   = filter.value["name"]
      values = filter.value["values"]
    }
  }
}

data "aws_instances" "nodes" {
  filter {
    name   = "tag:aws:autoscaling:groupName"
    values = module.eks.eks_managed_node_groups["gpu_node_pool"]["node_group_autoscaling_group_names"]
  }
  instance_state_names = ["running"]
}

data "aws_eks_cluster" "holoscan" {
  name = module.eks.cluster_name
  # role_arn = module.eks.cluster_arn
  # vpc_config {
  #   subnet_ids         = [var.existing_vpc_details == null ? module.vpc[0].private_subnets : var.existing_vpc_details.subnet_ids]
  # }
}
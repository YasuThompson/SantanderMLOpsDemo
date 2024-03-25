# from https://github.com/NVIDIA/nvidia-terraform-modules/blob/main/eks/main.tf and modified to fit the project

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      "Terraform" = "true"
      "class"     = "Study"
      "project"   = "mlflow-on-k8s-sample"
    }
  }
}


# Filter out local zones, which are not currently supported 
# with managed node groups
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

locals {
  cluster_name = "sample-eks-${random_string.suffix.result}"
}

resource "random_string" "suffix" {
  length  = 8
  special = false
}

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

/********************************************
  Network Config
********************************************/
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "4.0.2"
  count   = var.existing_vpc_details == null ? 1 : 0
  name    = "tf-${local.cluster_name}-vpc"
  cidr    = var.cidr_block
  azs     = data.aws_availability_zones.available.names
  # FUTURE: Make configurable, or set statically for the max number of pods a cluster can handle
  private_subnets         = var.private_subnets
  public_subnets          = var.public_subnets
  enable_nat_gateway      = var.enable_nat_gateway
  single_nat_gateway      = var.single_nat_gateway # Future: Revisit the VPC defaults
  enable_dns_hostnames    = var.enable_dns_hostnames
  map_public_ip_on_launch = true
}


/********************************************
  Kubernetes Cluster Configuration
********************************************/

locals {
  cluster_name = "sample-eks-${random_string.suffix.result}"

  holoscan_node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node ingress, no external ingress"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }

    egress_all = {
      description      = "Node egress to open internet"
      protocol         = "-1"
      from_port        = 0
      to_port          = 0
      type             = "egress"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = ["::/0"]
    }
  }
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.15.3"

  cluster_name    = local.cluster_name
  cluster_version = var.cluster_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  eks_managed_node_group_defaults = {
    ami_type = "AL2_x86_64"

  }

  eks_managed_node_groups = {
    cpu-node = {
      name = "node-group-cpu"

      instance_types = ["t3.small"]

      min_size     = 1
      max_size     = 3
      desired_size = 2
      capacity_type  = "SPOT"
     block_device_mappings = {
        root = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = var.node_pool_root_disk_size_gb
            volume_type           = var.node_pool_root_volume_type
            delete_on_termination = var.node_pool_delete_on_termination
          }
        }
      }

    }

    # gpu-node = {
    #   name = "node-group-gpu"

    #   instance_types = ["g4ad.xlarge"]
    #   # instance_types = ["t3.small"]

    #   min_size     = 1
    #   max_size     = 1
    #   desired_size = 1
    #   # capacity_type  = "SPOT"

    #   ami_id                     = var.gpu_iam_id
    #   ami_type                   = "CUSTOM"

    #   block_device_mappings = {
    #     root = {
    #       device_name = "/dev/xvda"
    #       ebs = {
    #         volume_size           = var.node_pool_root_disk_size_gb
    #         volume_type           = var.node_pool_root_volume_type
    #         delete_on_termination = var.node_pool_delete_on_termination
    #       }
    #     }
    #   }
    #   taints = {
    #     dedicated = {
    #       key    = "dedicated"
    #       value  = "gpuGroup"
    #       effect = "NO_SCHEDULE"
    #     }
    #   }

    # }
  }
}


# https://aws.amazon.com/blogs/containers/amazon-ebs-csi-driver-is-now-generally-available-in-amazon-eks-add-ons/ 
data "aws_iam_policy" "ebs_csi_policy" {
  arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

module "irsa-ebs-csi" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-assumable-role-with-oidc"
  version = "4.7.0"

  create_role                   = true
  role_name                     = "AmazonEKSTFEBSCSIRole-${module.eks.cluster_name}"
  provider_url                  = module.eks.oidc_provider
  role_policy_arns              = [data.aws_iam_policy.ebs_csi_policy.arn]
  oidc_fully_qualified_subjects = ["system:serviceaccount:kube-system:ebs-csi-controller-sa"]
}

resource "aws_eks_addon" "ebs-csi" {
  cluster_name             = module.eks.cluster_name
  addon_name               = "aws-ebs-csi-driver"
  addon_version            = "v1.20.0-eksbuild.1"
  service_account_role_arn = module.irsa-ebs-csi.iam_role_arn
  tags = {
    "eks_addon" = "ebs-csi"
    "terraform" = "true"
  }
}
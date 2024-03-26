# from https://github.com/NVIDIA/nvidia-terraform-modules/blob/main/eks/variables.tf and modified to fit the project

variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1" # Tokyo
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

variable "gpu_iam_id" {
  description = "value of the IAM role for the GPU node group"
  type        = string
  default     = "ami-0fe55d4bb80685904"
}

variable "node_pool_root_disk_size_gb" {
  type        = number
  default     = 512
  description = "The size of the root disk on all GPU nodes in the EKS-managed GPU-only Node Pool. This is primarily for container image storage on the node"
}


variable "node_pool_root_volume_type" {
  type        = string
  default     = "gp2"
  description = "The type of disk to use for the GPU node pool root disk (eg. gp2, gp3). Note, this is different from the type of disk used by applications via EKS Storage classes/PVs & PVCs"
}

variable "node_pool_delete_on_termination" {
  type        = bool
  default     = true
  description = "Delete the VM nodes root filesystem on each node of the instance type. This is set to true by default, but can be changed when desired when using the 'local-storage provisioner' and are keeping important application data on the nodes"
}

variable "cpu_node_pool_additional_user_data" {
  type        = string
  default     = ""
  description = "User data that is appended to the user data script after of the EKS bootstrap script on EKS-managed GPU node pool."
}
> [!WARNING]
> 
> AWS EKS clusters cost $0.10 per hour (and other components might cost more), so you may incur charges by running this tutorial. The cost should be a few dollars at most, but be sure to delete your infrastructure promptly to avoid additional charges. We are not responsible for any charges you may incur.



# 0. Prerequisites

- install awscli and setup awscli (`aws configure`).
- install `kubectl` command.

# 1. Create eks cluster 

Create EKS cluster using terraform file.

```
cd eks-terraform
terraform init 
terraform apply
export CLUSTER_ENDPOINT=$(terraform output cluster_endpoint)
export CLUSTER_ENDPOINT_HTTP=$(echo $CLUSTER_ENDPOINT | sed "s/https/http/")
```

Setup kubectl config.

```
aws eks --region $(terraform output -raw region) update-kubeconfig \
    --name $(terraform output -raw cluster_name)
```

Ensure that your EKS cluster is functioning properly.

```
kubectl get nodes -o wide
```

# 2. Test api 

## 2.1 Build api-test container

```
export DOCKEHUB_USER_NAME=<your dockerhub user name>
cd mlflow-api-test
docker build -t ${DOCKERHUB_USER_NAME}/mlflow-api-test:latest .
docker push ${DOCKERHUB_USER_NAME}/mlflow-api-test:latest
```

## 2.2 Create api-test deploy in EKS cluster

```
kubectl create deploy api-test --image ${DOCKERHUB_USER_NAME}/mlflow-api-test:latest --port 5000
kubectl expose deploy api-test --target-port 5000 --type=type=LoadBalancer
```

## 2.3 Check api-test deploy

Then check api-test deploy.
\<endpoint domain name\> is `terraform output `

```
curl http://${CLUSTER_ENDPOINT_HTTP}:5000/score
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"X": [1, 2]}'
# output
# {"score":[1,2]}
```

## 2.4 clean up 

Delete service and deploy. Ensure delete loadbalancder service before `terraform destroy`
```
kubectl delete svc api-test
kubectl delete deploy api-test
cd ../eks-terraform
terraform destroy
```

\- 終劇 -

## Reference

- kubernetes-mlops https://github.com/AlexIoannides/kubernetes-mlops/tree/master
- terraform eks tutorial https://developer.hashicorp.com/terraform/tutorials/kubernetes/eks

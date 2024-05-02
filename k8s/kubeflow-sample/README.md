# ML pipline using Kubeflow

[!NOTE] This document is work in progress

## 0.Prerequisite

- Kubernetes Cluster
- Kubeflow
- kfp (use pip to install)
- mssql server

[!NOTE] In this document, we deployed kubernetes cluster using [microk8s](https://microk8s.io) and kubeflow using [juju](https://charmed-kubeflow.io/docs/install). You shuld change user name:`admin` and password:`admin`, kubeflow endpoint:`10.64.140.43.nip.io` in `component/clean_data.py` (those are default user name and password, kubeflow endpoint url of kubeflow deployed with juju).

## 1. Set your enviroment variable

You need to export below variables.

- MSSQL_SERVER: mssql server address (like AAA.BBB.windows.net)
- MSSQL_PASSWORD: mssql server login user
- MSSQL_USER: msssql server login password
- MSSQL_DATABASE: database name

## 2. Set up docker image for pipline

```bash
cd docker
docker build -t "${DOCKER_REGISTRY}/kubeflow-images/mssql-cmd:latest" -f "msssql-dockerfile"  --platform linux/amd64 --no-cache
docker push "${DOCKER_REGISTRY}/kubeflow-images/mssql-cmd:latest"
```

[!NOTE] `DOCKER_REGISTRY` is private (or public) docker registry. If you use microk8s on your local machine and enable microk8s registry addon, it shuld be `localhost:32000`. And ensure your registry shuld support https connection or your cluster can reach your registry using http connection.

[!NOTE]　By building this docker image, you agree to the mssql EULA.

## 3. Run pipline

```bash
cd component
python clean_data.py
```

-終劇-

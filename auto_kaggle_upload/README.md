# SantanderMLOpsDemo/auto_kaggele_upload

## 方法1. 手動でコンテナを起動する．

### docker image(ham_mlops_prototypeにて)

```sh
docker build -t kaggle_upload ./auto_kaggle_upload
```

### kaggle.jsonの設定(ham_mlops_prototypeにて)

```sh
mkdir data
cp path_to_json/kaggle.json data/
chmod 600 data/kaggle.json
```

### docker container(ham_mlops_prototypeにて)

```sh
docker run --rm -d -v $(pwd):/work mlops:kaggle_upload_test2 sh auto_kaggle_upload/submit.sh
```

## 方法2. 自動でコンテナを起動する．(docker composeを使う)

### docker compose でコンテナを起動する．

下のコマンドで(多分docker imageのビルドを含めて)kaggleに送信するためのコンテナを立ち上げる．

window, MAC(??)の場合

```sh
docker compsoe up -d
```

linuxの場合

```sh
docker-compose up -d
```

起動しない場合，もう片方のコマンドで起動するとうまく行く可能性がある．

ここでは，dataディレクトリ下にkaggle.jsonが配置されていると仮定している．

buildが自動で行われないとき上記コマンドのup -dの代わりにbuildを用いてイメージを作成する．

```sh
docker compose build
```

## 参考

* [Kaggle apiとPythonを使って、サブミットをする方法](https://www.currypurin.com/entry/kaggle-api-submit)
* [Kaggle github repository](https://github.com/Kaggle/kaggle-api)
<!-- *  -->
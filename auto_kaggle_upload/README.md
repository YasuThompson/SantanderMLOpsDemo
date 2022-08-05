# ham_mlops_prototype/auto_kaggele_upload

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


### 参考：
* [Kaggle apiとPythonを使って、サブミットをする方法](https://www.currypurin.com/entry/kaggle-api-submit)
* [Kaggle github repository](https://github.com/Kaggle/kaggle-api)
<!-- *  -->
# Tigress Build Script

## IDA_script

You can generate IDB by passing binary into IDA


Dataset Generation
```
python IDA_script/dataset_creation.py -d (path-to-idb-directory)
```


## dataset_buildscript

Tigress Build Script (tested for coreutils-8.2)
```
python dataset_buildscript/buildscript.py -d (path-to-directory-containing-packages)
```

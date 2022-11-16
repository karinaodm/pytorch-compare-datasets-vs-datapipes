# Comparing PyTorch Dataset and TorchData DataPipes

## Prepare dataset
* Load aligned CelebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* Load DigiFace1M dataset from https://github.com/microsoft/DigiFace1M
```bash
data
├── CelebA
│   ├── identity_CelebA.txt
│   ├── img_align_celeba
│       ├── 000001.jpeg
│       ├── 000002.jpeg
...
├── DigiFace1M
    ├── 0
    |   ├── 0.png
    |   ├── 1.png
     ...
    ├── 1
        ├── 0.png
        ├── 1.png
     ...
```
## Run experiments
* Run docker
```
./build-docker.sh
docker run -it --gpus all --ipc=host --rm -v `pwd`:/working_dir data_load_example:latest
```
* In docker container run
```
python3 src/dataset_example.py
python3 src/datapipe_example.py
python3 src/datapipe_example2.py --prepare_data
```
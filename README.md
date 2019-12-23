### Build pytorch docker with 1.3.0, 1.3.1 (CUDA 10.1)
```
sudo docker build -f Dockerfile.pytorch --build-arg PYTORCH_VER=1.3.0 -t test/test:torch130_cu101 .
sudo docker build -f Dockerfile.pytorch --build-arg PYTORCH_VER=1.3.1 -t test/test:torch131_cu101 .
```

Run pytorch docker and make .onnx outputs with torch2onnx.py.
```
sudo docker run -v$(pwd):/home/ --gpus all test/test:torch130_cu101 python /home/torch2onnx.py -s /home/torch130_out
sudo docker run -v$(pwd):/home/ --gpus all test/test:torch131_cu101 python /home/torch2onnx.py -s /home/torch131_out
```

### Build tensorrt 7.0.0. docker (CUDA 10.2)
Install trt .deb file before build TRT docker.
```
sudo docker build -f Dockerfile.trt -t test/test:TRT700 .
```

Run TRT docker and make engine with onnx2engine.py.
```
sudo docker run -v$(pwd):/home/ --gpus all test/test:TRT700 python3 /home/onnx2engine.py --onnx_folder /home/torch130_out
sudo docker run -v$(pwd):/home/ --gpus all test/test:TRT700 python3 /home/onnx2engine.py --onnx_folder /home/torch131_out
```

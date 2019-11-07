#Create a conda env.

`conda create -n ucx -c conda-forge -c jakirkham/label/ucx-new cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7`

# Install some other dependencies

```
conda install -c sarcasm fmt
conda install -c conda-forge spdlog
conda install -c conda-forge gflags
conda install -c conda-forge cppzmq
conda install -c anaconda boost

```

# Compile
``
./conda/recipes/blazingdb-communication/build.sh
``

# How to use
`./ucx_benchmark -device_id_client=0 -device_id_server=1 -context=tcp`



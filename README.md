# SCAGE

## Install

```bash
git clone <this_repo> scage

cd scage

python3.10 -m venv .venv

source .venv/bin/activate

pip install --editable ".[latptop]"
```

Run with GPU, for example on ASCAD trimmed:
```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH

scage -d f-opoi -c example/sca.yml -g example/sca.grammar -o /tmp/scage_test -r 1
```

Names for valid dataset are available in the `dataset_parameters.py` file.

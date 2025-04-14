# Setup ASCAD fixed

Get ASCAD fixed dataset as instructed [here](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key).

## Instructions

```bash
# Download dataset
wget https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip
unzip ASCAD_data.zip

# Move .h5 files into current folder
mv -t . ASCAD_data/ASCAD_databases/*
# Remove unnecessary files
rm -r ASCAD_data ASCAD_data.zip
```

After this you can use the `generate_dataset.py` to resample raw traces and apply desynchronization.

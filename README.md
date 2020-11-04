# cloths_segmentation
Code for binary segmentation of cloths

## Installation

`pip install -U cloths_segmentation`

### Example inference

Jupyter notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18RenTYhuPVip9SHdMLn-vnK0K57B--um#scrollTo=D0h2Y-oOCnXJ)

## Data Preparation

Download the dataset from [https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6)

Process the data using script [https://github.com/ternaus/iglovikov_helper_functions/tree/master/iglovikov_helper_functions/data_processing/prepare_cloths_segmentation](https://github.com/ternaus/iglovikov_helper_functions/tree/master/iglovikov_helper_functions/data_processing/prepare_cloths_segmentation)

The script will create process the data and store images to folder `images` and binary masks to folder `labels`.

## Training

### Define the config.
Example at [cloths_segmentation/configs](cloths_segmentation/configs)

You can enable / disable datasets that are used for training and validation.

### Define the environmental variable `IMAGE_PATH` that points to the folder with images.
Example:
```bash
export IMAGE_PATH=<path to the the folder with images>
```

### Define the environmental variable `LABEL_PATH` that points to the folder with masks.
Example:
```bash
export MASK_PATH=<path to the folder with masks>
```

### Training
```
python -m cloths_segmentation.train -c <path to config>
```

### Inference

```bash
python -m torch.distributed.launch --nproc_per_node=<num_gpu> cloths_segmentation/inference.py \
                                   -i <path to images> \
                                   -c <path to config> \
                                   -w <path to weights> \
                                   -o <output-path> \
                                   --fp16

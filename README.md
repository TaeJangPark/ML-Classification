# Multi Label Classification
This is an implemtation of Multi Label Classification using Tensorflow. The model generates class names and scores about an image.

The repository includes :
* Source code of Multi Label Classificatio on MobileNet V2.
* Training code for Open Image Dataset V4.
* Jupyter notebook to visualize the recognition results.

# Getting Strarted
* ([train.py](train.py), [mobilenet_v2.py](/networks/mobilenet_V2.py), [config.py](/configs/config.py), [datapipe.py](/data/datapipe.py)): These files contain the Multi Label Classification implementation.

*[train_class_name.csv](/data/train_class_name.csv]: This file contains the class names and class index for Open Image Dataset in train folder.

* [inspect_model.ipynb](/inspect_model.ipynb) This notebook provides visualizations of every step of the pipeline for recognizing multi label on image.

# Training on OIDv4
you need OID_V4(Open Image Dataset V4) and put the dataset in data/OIDv4, if you want to train this model.
Run train.py


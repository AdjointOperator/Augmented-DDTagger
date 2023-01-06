# Augmented DeepDanbooru Tagger

## What is this? Why does it ever exist?

The original DeepDanbooru is a great tool, but it has some performance limitations. This project is an attempt to improve the performance of the original DeepDanbooru by using batched processing of images and multi-threading dataloader. The wd-1.4 ConvNext tagger is also supported. It also comes with useful functionalities like limiting the total number of CLIP tokens it splits out for applications like StableDiffusion based model fine-tuning with scal-sdt, naifu-diffusion, or A1111's webui's DreamBooth.

## How to use it?

1. Install the requirements.
2. Download the corresponding models and put them in the `models` folder: `models/deepbooru` for DeepDanbooru and `models/wd-v1-4-convnext-tagger` for ConvNext (recommended; specify the model path otherwise). The models can be downloaded from the original repos ([Deepdanbooru](https://github.com/KichangKim/DeepDanbooru/releases) and [ConvNext](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger/tree/main)).
3. Configure the config file. The content should be rather straight-forward, and every parameter can be overrode by command line arguments. Please refer to the comments in the config file for more details.
4. `python3 main.py -c path_to_your_config.yaml [-d PATH_TO_FOLDER]`

## Cautions

1. The program will not finish if there is any corrupted image in the folder. Please remove them before using the program, or just kill the program and restart it after the corrupted image is removed.
2. The prediction is cached in the hdf5 file created in the same folder as the images, and the relative path of the image w.r.t. the folder is used as the key. If you are replacing images in the folder without changing the relative path, the out-dated prediction will be used. New predictions will be added to the cache file and cache prediction without corresponding images will be removed. If you want to force the prediction to be re-calculated, please delete the cache file.

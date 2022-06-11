# Universal GANs starter

This is an universal GANs starter for image generation models. It is an convinient instrument for training models. You can easily add new generators, losses, regularizations and many other parts. If you have a specific model, you dont need to change all code, but just specific parts. All logs by default saving at [wandb](https://wandb.ai).

This starter is based on [Style GAN 2](https://github.com/NVlabs/stylegan2) training code.

## Prepaired configs
- [x] DCGAN
- [x] WGAN-GP
- [x] SN-GAN 
- [x] Style GAN 2 ADA
- [x] Big GAN

## Requirements
- Install python packages
    ```bash
    pip install -r requirements.txt
    ```

## Preparing datasets
To setup dataset you should use ```stylegan2ada/dataset_tool.py``` as it described in [Style GAN 2 ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) repository.



## Training
To train network you should run following command:
```bash
    python starter.py exp.config_dir=configs/ exp.config=some_config.yaml exp.name=exp_name
```
Starter use a omegaconf package, so it is very convinient to change some of the parametrs without changin config. For example, if you want to change batch size, you can add ```gen.batch=50``` to the command line.

## Adding new parts
For example, you want to add new generator architecture. You need to add class of this architecture in ```training_parts/generators.py``` and decorate it by ```@generators.add_to_registry("name")```, where "name" will be used as key in config for this generator. 

Similarly with other training parts.
    




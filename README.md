# EMB96: Endless Music Box 96

EMB96 is an AI (Artificial Intelligence) trained to generate midi.


## Sound Representation

Each midi file is represented as a 16 piano-roll composed of 96 notes and 96 values.
Why 96:
  * 8 octaves
  * devides almost every scale


## Model

The model is a simple VAE (variationnal Auto Encoder) trained to reconstruct a dataset of midi files.
The networks tries to encode the data into a 128 vector $z$ (variable from latent space) and then decodes this vector to reconstruct the original data.
It has been trained using a mixture of the KLD (kl divergence) fot the sparsity of the latent space and the binary cross entropy loss for the reconstruction part.


## How to use it

### Build dataset

First of all you need midi files. You can use your own files or you could choose to use the same dataset we used as an example.

First, place you data into a single directory.
To donwload the example dataset:

```bash
cd src
python3 main.py --dataset_dir <dataset dir path> --donwload
```

Then you need to convert these midi files as images to feed the network:

```bash
cd src
python3 main.py --dataset_dir <dataset dir path> --build_dir <build dir path> --build
```

## Train model

**For the moment you can only train on gpu**

You can train the model just by calling the following command:

```bash
cd src
python3 main.py --build_dir <build dir path> --train
```

If you want to change the training parameters you can. You can access the hole list of parameters using the following:

```bash
cd src
python3 main.py --help
```

## Test the model

**This part is accessible for non gpu users, you just have to use the option --cpu**

If you want to try the model:

```bash
cd src
python3 main.py --checkpoint <checkpoint path> --n_examples <number of generated midi files> --output_dir <output directory> --test
```

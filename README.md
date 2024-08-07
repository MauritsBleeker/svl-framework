[![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](https://arxiv.org/abs/2402.17510)

# Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning

This repository contains the code for the TMLR paper [Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning](https://arxiv.org/abs/2402.17510),  by [Maurits Bleeker](https://mauritsbleeker.github.io)<sup>1</sup>, [Mariya Hendriksen](https://mariyahendriksen.github.io)<sup>1</sup>, [Andrew Yates](https://andrewyates.net)<sup>1</sup>, and [Maarten de Rijke](https://staff.fnwi.uva.nl/m.derijke/)<sup>1</sup>.

The implementation builds upon the codebase of [Latent Target Decoding](https://github.com/MauritsBleeker/reducing-predictive-feature-suppression/).

<sup>1</sup>University of Amsterdam, The Netherlands

## News
 - Jul 2024: The paper has been accepted by TMLR
 - Feb 2024: Initial arXiv release

## Requirements

To set up the environment, install the requirements using the provided YAML file:

```angular2html
conda env create --file src/environment.yaml
```

This command will create a conda environment `contrastive-shortcuts`. Activate the created environment:

```angular2html
source activate contrastive-shortcuts
```

## Training the models 

For local development, execute the following command:

```angular2html
python src/trainer.py --yaml_file src/configs/{f30k, coco}/development_local.yaml
```

To train a model run `python src/trainer.py` and provide a base config in YAML format using `--yaml_file <config path.yaml>`. 

Hyperparameters can be overridden using command line flags. For example:
```angular2html
python src/trainer.py --yaml_file src/configs/f30k/development_local.yaml --experiment.wandb_project <your project name>
```

The recommended approach is to have a fixed base config for each experiment and only modify specific hyperparameters for different training/evaluation settings. 

All training and evaluation were conducted using a SLURM-based scheduling system.

### Data loading and preparation

We implemented a PyTorch Dataloader class that loads the images from the memory of the compute node the training runs on. The captions are loaded from either the Flickr30k or MS-COCO annotation file.

Update the *.yaml config with the right file paths.

```angular2html
img_path:
annotation_file:
annotation_path:
```


### Vocabulary class

To create the vocabulary class, run:

```angular2html
python utils/vocab.py 
```
With the appropriate input flags.


### Job files

Job and hyperparameter files to reproduce experiments can be found in `src/jobs/{coco, f30k}/`.

The shortcut experiments (Section 4) are available in the `shortcuts` folder, the LTD experiments in the `LTD` folder, and the IFM experiments in the 'IFM' folder (Section 6).

## Evaluation

To reproduce results from Section 3, run the following evaluation script (ensure correct file paths).

```angular2html
sbatch src/jobs/{coco, f30k}/snellius/shortcuts/{clip, vse}/{clip, vse}_{coco, f30k}_shortcut_experiments_eval.job
```

Next, copy all the RSUM values to `notebooks/visualizations/visualization.ipynb` to generate the plot.

The results from Section 6 are generated by using `notebooks/Evaluation.ipynb`.

## Citing and Authors
If you find this repository helpful, feel free to cite our paper "[Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning](https://arxiv.org/abs/2402.17510)":

```latex
@article{bleeker-2024-demonstrating,
  title={Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning},
  author={Bleeker, Maurits and Hendriksen, Mariya and Yates, Andrew and de Rijke, Maarten},
  journal={Transactions on Machine Learning Research},
  url={https://openreview.net/forum?id=gfANevPraH},
  year={2024}
}
```

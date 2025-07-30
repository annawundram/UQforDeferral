# Is UQ a Viable Alternative to Learned Deferral?
This repository contains the code needed to recreate the results in the work titled "Is UQ a Viable Alternative to Learned Deferral?".

## Virtual Environment Setup

The code is implemented in Python 3.12.8. All required packages can be found in ```environment.yml```.

## Data
All experiments were conducted on data from the [AIROGS](https://ieeexplore.ieee.org/abstract/document/10253652) challenge. We preprocessed the dataset using the cropping function in [this](https://github.com/berenslab/fundus_image_toolbox) package. To split the dataset and save it to an H5 file run ```data/AIROGS_to_h5.py```. To create a blurry or noisy out-of-distribution dataset run ```data/create_ood_dataset.py```. The dataloaders for the in-distribution and out-of-distribution dataset can be found in ```data/AIROGS_dataloader.py``` and ```data/AIROGS_ood_dataloader.py``` respectively. The classification performance results on the [Chákṣu](https://www.nature.com/articles/s41597-023-01943-4) data were computed on images captured by the Bosch device. For preprocessing and dataloader please refer to [this](https://github.com/annawundram/glaucoma-diagnosis-pipeline) repository.

## Models
We compared seven different models in this work.

**Softmax:** To train the so called Softmax method, train a single ResNet50 using ```models/softmax_ensemble_mcdropout_LD1/train.py``` with ```--type=default```.

**Ensemble:** Train ten ResNet50s with different seeds (0, ..., 9) using ```models/softmax_ensemble_mcdropout_LD1/train.py``` with ```--type=default``` and set ```--random-seed``` accordingly.

**MC Dropout:** For a ResNet50 with dropout, train a ResNet50 with permanent dropout using ```models/softmax_ensemble_mcdropout_LD1/train.py``` with ```--type=dropout```.

**BNN:** To train a Bayesian ResNet50 use the script ```models/BNN/main_bayesian_flipout.py```. For more details on the implementation please visit [this](https://github.com/IntelLabs/bayesian-torch/tree/main) repo from which this code has been adapted.

**SWAG:** Train a SWAG model by running ```models/SWAG/train_swag.py```. For more details, please refer to the original implementation [here](https://github.com/wjmaddox/swa_gaussian) from which this code has been adapted.

**Learned Deferral one-stage:** Use the script ```models/softmax_ensemble_mcdropout_LD1/train.py``` with ```--type=defer``` and set the deferral cost ```--c``` accordingly. This will automatically employ the surrogtate loss function during training and adjust the number of output classes.

**Learned Deferral two-stage:** To train a two-stage Learned Deferral model, first an Ensemble must be trained as its outputs serve as the input to the second stage. This second-stage input is computed by running ```models/ensemble_outputs.py```. The second stage can then be trained using the script ```models/train_liu_deferral.py```.

To perform inference and save model outputs for the test set (both in-domain and out-of-domain), use the following scripts based on the model:
- Softmax model, the Ensemble and the one-stage Learned Deferral model: Run ```models/softmax_ensemble_mcdropout_LD1/eval_models.py```
- MC Dropout: Run ```models/softmax_ensemble_mcdropout_LD1/eval_mcdropout.py```
- BNN: Run ```models/BNN/main_bayesian_flipout.py``` with ```--mode=test```
- SWAG: Run ```models/SWAG/swag_eval.py```
- Two-stage Learned Deferral: Run ```models/LD2/eval_LD2.py```

## Plots
To evaluate all models and create the plots used in the paper, run the Notebook ```plots/evaluation.ipynb```.

# Is UQ a Viable Alternative to Learned Deferral?
This repository contains the code needed to recreate the results in the work titled "Is UQ a Viable Alternative to Learned Deferral?".

## Data
All experiments were conducted on data from the [AIROGS](https://ieeexplore.ieee.org/abstract/document/10253652) challenge. To preprocess this dataset and save it to an H5 file run ```data/AIROGS_to_h5.py```. To create a blurry or noisy out-of-distribution dataset run ```data/create_ood_dataset.py```. The dataloaders for the in-distribution and out-of-distribution dataset can be found in ```data/AIROGS_dataloader.py``` and ```data/AIROGS_ood_dataloader.py``` respectively.

## Models
We compared seven different models in this work.

**Softmax:** To train the so called Softmax method, train a single ResNet50 using ```models/softmax_ensemble_mcdropout_LD1/train.py``` using ```type=default```.

**Ensemble:** Train ten ResNet50s with different seeds (0, ..., 9) using ```models/softmax_ensemble_mcdropout_LD1/train.py``` using ```--type=default``` and set ```--random-seed``` accordingly.

**MC Dropout:** For a ResNet50 with dropout, train a ResNet50 with permanent dropout using ```models/softmax_ensemble_mcdropout_LD1/train.py``` using ```type=dropout```.

**BNN:** To train a Bayesian ResNet50 use the script ```models/BNN/main_bayesian_flipout.py```. For more details on the implementation please visit [this](https://github.com/IntelLabs/bayesian-torch/tree/main) repo from which it has been adapted.

**SWAG:** Train a SWAG model by running ```models/SWAG/train_swag.py```. For more details, please refer to the original implementation [here](https://github.com/wjmaddox/swa_gaussian) from which this has been adapted.

**Learned Deferral one-stage:** Use the script ```models/softmax_ensemble_mcdropout_LD1/train.py``` using ```--type=defer``` and set the deferral cost ```--c``` accordingly. This will automatically employ he surrogtate loss function during training and adjust the number of output classes.

**Learned Deferral two-stage:** To train a two-stage Learned Deferral model, first an Ensemble as its outputs serve as the input to the second stage. This input is computed by running ```models/ensemble_outputs.py```. The second stage can then be trained using the script ```models/train_liu_deferral.py```.

To perform inference and save model outputs for the test set (in- and out-of-domain) for the Softmax model, the Ensemble and the one-stage Learned Deferral model, run ```models/softmax_ensemble_mcdropout_LD1/eval_models.py```. For MC Dropout use ```models/softmax_ensemble_mcdropout_LD1/eval_mcdropout.py```. For the two-stage Learned Deferral use ```models/LD2/eval_LD2.py```. For BNN run ```models/BNN/main_bayesian_flipout.py``` with ```--mode=test```. For SWAG run ```models/SWAG/swag_eval.py```.

## Plots
To evaluate all models and create the plots used in the paper, run the Notebook ```plots/evaluation.ipynb```.

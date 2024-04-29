# Next Activity Prediction and Uncertainty

Code for the chapter "*Next Activity Prediction and Uncertainty*" in the PhD Thesis "*Machine Learning for Probabilistic and Attribute-Aware Process Mining*".
The libraries required are in the "*requirements.txt*" file. To install with pip:

```
pip install requirements.txt
```
 
Four datasets are available: 
    - *Business Process Intelligence 2012* -> *bpic2012*
    - *Business Process Intelligence 2013* -> *bpic2013*
    - *Helpdesk* -> *helpdesk*
    - *Road Traffic Fine Management Process* -> *road_traffic_fines* [to be unzipped first since it was too big for github]

The main function admits two modalities: training and visualization. In both modalities the input must include the dataset required.

## Training

For example, to train the models on the BPIC 2012 dataset:

```
python main.py bpic2012 train
```

## Visualization

In the visualization mode, other parameters are avaliable:

    - --model-dir: directory where the models are stored [mandatory]. In the directory "*models*", each dataset has its own subdirectory. The models reproducing the chapter results are under the directory "*ensemble_1*"
    - --uncertainty_type: how the uncertainty is injected in the prediction [optional] -> *ensemble* = ensemble of models, *MC*= Monte Carlo dropout. Default MC.
    - --predict_case: id of the specific case to be predicted [optional]. Default None.
    - --overconfidence_analysis: boolean to indicate whether running the analysis on the overconfident cases [optional]. Default False.

For example, to visualize the results on BPIC 2012 dataset, with stochasticity given by an ensemble of models and overconfident analysis:

```
python main.py bpic2012 vis --model_dir ensemble_1 --uncertainty_type ensemble --overconfidence_analysis True
```

Numerical results are stored in the same directory of the models while figures are in the *figures* directory. 

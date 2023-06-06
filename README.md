# Robust Detection Outcome (RoDeO)

<div align="center">

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rodeometric)](https://pypi.org/project/rodeometric/)
[![PyPI Status](https://badge.fury.io/py/rodeometric.svg)](https://badge.fury.io/py/rodeometric)
[![Conda](https://img.shields.io/conda/v/conda-forge/rodeometric?label=conda&color=success)](https://anaconda.org/conda-forge/rodeometric)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/FeliMe/RoDeO/blob/main/LICENSE)

</div>

______________________________________________________________________

Official Repository of [Robust Detection Outcome: A Metric for Pathology Detection in Medical Images](https://openreview.net/forum?id=zyiJi4sJ7dZ).
RoDeO is an easy to use object detection metric useful for (but not limited to) applications in medical imaging, such as pathology detection in Chest X-ray images.
It evaluates three sources of errors (**misclassification, faulty localization, and shape mismatch**) separately and combines them to one score.
RoDeO better fulfills requirements in medical applications through its **interpretability**, notion of **proximity** and strong penalization of over- and under-prediction, **encouraging precise models**.

![Title Figure](https://github.com/FeliMe/RoDeO/blob/6eefdc43b62c798f2925aee65876d4eb788c14d3/assets/title_figure.png?raw=true)

# Installation

RoDeO is available as a python package for python 3.7+ as [rodeometric](https://pypi.org/project/rodeometric/). To install, simply install it with pip:
```shell
python -m pip install rodeometric
```

# Usage

```python
import numpy as np
from rodeo import RoDeO

# Init RoDeO with two classes
rodeo = RoDeO(class_names=['a', 'b'])
# Add some predictions and targets
pred = [np.array([[0.1, 0.1, 0.2, 0.1, 0.0],
                  [0.0, 0.3, 0.1, 0.1, 1.0],
                  [0.2, 0.2, 0.1, 0.1, 0.0]])]
target = [np.array([[0.0, 0.0, 0.1, 0.1, 0.0],
                    [0.0, 0.2, 0.1, 0.1, 1.0]])]
rodeo.add(pred, target)
# Compute the score
score = rodeo.compute()
for key, val in score.items():
    print(f'{key}: {val}')
```

# Advantages of RoDeO

1. AP@IoU benefits from severe overprediction at higher thresholds

 RoDeO | AP@IoU | acc@IoU
:-----:|:------:|:-------:
![Overprediction RoDeO](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_overperclass_fixedsizesigma_RoDeO.png?raw=true) | ![Overprediction AP@IoU](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_overperclass_fixedsizesigma_AP.png?raw=true) | ![Overprediction acc@IoU](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_overperclass_fixedsizesigma_acc.png?raw=true)

2. Acc@IoU achieves high scores with underprediction due to the dominance of true negatives

 RoDeO | AP@IoU | acc@IoU
:-----:|:------:|:-------:
![Underprediction RoDeO](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_undersample_fixedsizesigma_RoDeO.png?raw=true) | ![Underprediction AP@IoU](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_undersample_fixedsizesigma_AP.png?raw=true) | ![Underprediction acc@IoU](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_undersample_fixedsizesigma_acc.png?raw=true)

3. Compared to threshold-based metrics (like Average Precision @ IoU), RoDeO degrades more gracefully and has a better notion of proximity

![Localation error RoDeO](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_randcorrupt_relpossize_RoDeO.png?raw=true)
 AP@IoU | acc@IoU
:------:|:-------:
![Localation error AP@IoU](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_randcorrupt_relpossize_AP.png?raw=true) | ![Localation error acc@IoU](https://github.com/FeliMe/RoDeO/blob/babef650894f8eacc82a2f23ac69997cab13d39d/assets/boxoracle_randcorrupt_relpossize_acc.png?raw=true)

<!-- # Citation
If you use RoDeO in your project, please cite
```
@inproceedings{rodeo-midl2023,
  author    = {Felix Meissen and Philip Müller and Georgios Kaissis and Daniel Rückert},
  title     = {Robust Detection Outcome: A Metric for Pathology Detection in Medical Images.},
  booktitle = {MIDL},
  year      = {2023},
}
``` -->
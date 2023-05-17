# CNNs vs. Transformers: Performance, Robustness and Generalization in Gastrointestinal Endoscopic Image Analysis
 
 This repository contains the codebases for the following publication(s):
 - Carolus H.J. Kusters *et al.* - CNNs vs. Transformers: Performance and Robustness in Endoscopic Image Analysis  *(Submission under Review)*
 - Carolus H.J. Kusters *et al.* - Will Transformers transform endoscopic AI? A comparative analysis between CNNs and Transformers, in terms of performance, robustness and generalization *(Submission under Review)*

## Folder Structure
The folder structure of this repository is as follows:

```bash
├── data
│   ├── dataset_public.py
│   └── dataset_wle.py
├── models
│   ├── CaraNet.py
│   ├── ConvNeXt.py
│   ├── ESFPNet.py
│   ├── FCBFormer.py
│   ├── ResNet.py
│   ├── SwinUperNet.py
│   ├── UNet.py
│   └── model_wle.py
├── preprocess
│   ├── generate_cache.py
│   └── generate_cache_public.py
├── utils
│   ├── loss_optim_wle.py
│   └── metrics_wle.py
├── pretrained
│   └── ...
├── conda.yaml
├── inference_public.py
├── inference_wle.py
├── train_public.py
├── train_wle
└── README.md
```

## Environment
The conda environment with the dependencies for this project are specified in "conda.yaml".

## Pretrained models
The ImageNet pretrained weights are downloaded from the following links, and put under the "pretrained" folder: 
- CaraNet (Res2Net): https://github.com/Res2Net/Res2Net-PretrainedModels
- MiT: https://github.com/dumyCq/ESFPNet
- PvTv2: https://github.com/whai362/PVT/tree/v2/classification
- SwinV2: https://github.com/microsoft/Swin-Transformer

## Model Comparison


## Results

## Citation
If you think this helps, please use the following citation:

```bash
@inproceedings{KustersConference,
title = {{CNNs vs. Transformers: Performance and Robustness in Endoscopic Image Analysis}},
author = {Kusters, Carolus H.J. and Boers, Tim G.W. and Jaspers, Tim J.M. and Jong, Martijn R. and Jukema, Jelmer B. and de Groof, Albert J and Bergman, Jacques J and van der Sommen, Fons and de With, Peter H.N.},
booktitle = {Submission under review},
year = {2023}
}

@article{KustersJournal,
title = {Will Transformers transform endoscopic AI? A comparative analysis between CNNs and Transformers, in terms of performance, robustness and generalization},
journal = {Submission Under Review},
volume = {},
pages = {},
year = {2023},
issn = {},
doi = {},
url = {},
author = {Kusters, Carolus H.J. and Boers, Tim G.W. and Jaspers, Tim J.M. and Jong, Martijn R. and Jukema, Jelmer B. and de Groof, Albert J and Bergman, Jacques J and van der Sommen, Fons and de With, Peter H.N.},
keywords = {Endoscopic Image Analysis, Convolutional Neural Networks, Transformers, Robustness, Generalization},
abstract = {}
}
```

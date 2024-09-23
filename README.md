<div align="center">
<h1>Publications</h1>
<h3>CNNs vs. Transformers: Performance and Robustness in Endoscopic Image Analysis</h3>

[Carolus H.J. Kusters](https://chjkusters.github.io/)<sup>1 :email:</sup>, [Tim G.W. Boers](https://scholar.google.nl/citations?user=_TdckGAAAAAJ&hl=nl&oi=ao)<sup>1</sup>, [Tim J.M. Jaspers](https://scholar.google.nl/citations?user=nwfiV2wAAAAJ&hl=nl&oi=ao)<sup>1</sup>, Jelmer B. Jukema<sup>2</sup>, [Martijn R. Jong](https://scholar.google.nl/citations?user=QRNrL-oAAAAJ&hl=nl&oi=ao)<sup>2</sup>, Kiki N. Fockens<sup>2</sup>, Albert J. de Groof<sup>2</sup>, [Jacques J. Bergman](https://scholar.google.nl/citations?user=4SFBE0IAAAAJ&hl=nl&oi=ao)<sup>2</sup>, [Fons van der Sommen](https://scholar.google.nl/citations?user=qFiLkCAAAAAJ&hl=nl&oi=ao)<sup>1</sup>, Peter H.N. de With<sup>1</sup>
 
<sup>1</sup>  Department of Electrical Engineering, Video Coding & Architectures, Eindhoven University of Technology <br /> <sup>2</sup>  Department of Gastroenterology and Hepatology, Amsterdam University Medical Centers, University of Amsterdam

(<sup>:email:</sup>) corresponding author

*Second Workshop on Applications of Medical AI &#40;AMAI&#41; - Satellite Event MICCAI 2023* <br /> ([Proceeding](https://link.springer.com/chapter/10.1007/978-3-031-47076-9_3))



<h3>Will Transformers transform gastrointestinal endoscopic image analysis? A comparative analysis between CNNs and Transformers, in terms of performance, robustness and generalization</h3>

[Carolus H.J. Kusters](https://chjkusters.github.io/)<sup>1 :email:</sup>, [Tim J.M. Jaspers](https://scholar.google.nl/citations?user=nwfiV2wAAAAJ&hl=nl&oi=ao)<sup>1</sup>, [Tim G.W. Boers](https://scholar.google.nl/citations?user=_TdckGAAAAAJ&hl=nl&oi=ao)<sup>1</sup>, Jelmer B. Jukema<sup>2</sup>, [Martijn R. Jong](https://scholar.google.nl/citations?user=QRNrL-oAAAAJ&hl=nl&oi=ao)<sup>2</sup>, Kiki N. Fockens<sup>2</sup>, Albert J. de Groof<sup>2</sup>, [Jacques J. Bergman](https://scholar.google.nl/citations?user=4SFBE0IAAAAJ&hl=nl&oi=ao)<sup>2</sup>, [Fons van der Sommen](https://scholar.google.nl/citations?user=qFiLkCAAAAAJ&hl=nl&oi=ao)<sup>1</sup>, Peter H.N. de With<sup>1</sup>
 
<sup>1</sup>  Department of Electrical Engineering, Video Coding & Architectures, Eindhoven University of Technology <br /> <sup>2</sup>  Department of Gastroenterology and Hepatology, Amsterdam University Medical Centers, University of Amsterdam

(<sup>:email:</sup>) corresponding author

*Medical Image Analysis - Volume 99, January 2025* <br /> ([Article](https://doi.org/10.1016/j.media.2024.103348))

</div>


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
The conda environment with the software dependencies for this project are specified in "conda.yaml".

## Pretrained models
The ImageNet pretrained weights are downloaded from the following links, and put under the "pretrained" folder: 
- CaraNet (Res2Net): https://github.com/Res2Net/Res2Net-PretrainedModels
- MiT: https://github.com/dumyCq/ESFPNet
- PvTv2: https://github.com/whai362/PVT/tree/v2/classification
- SwinV2: https://github.com/microsoft/Swin-Transformer

## Model Comparison
The study aims to comprehensively evaluate and compare the peak performance, generalization, robustness and complexity of several CNNs and Transformers for gastrointestinal endoscopic image analysis. The following models are involved in the comparison.
![My Image](images/Models.PNG)

## Citation

If you think this helps, please use the following citation for the conference paper:
```bash
@InProceedings{10.1007/978-3-031-47076-9_3,
author="Kusters, Carolus H. J.
and Boers, Tim G. W.
and Jaspers, Tim J. M.
and Jukema, Jelmer B.
and Jong, Martijn R.
and Fockens, Kiki N.
and de Groof, Albert J.
and Bergman, Jacques J.
and van der Sommen, Fons
and de With, Peter H. N.",
editor="Wu, Shandong
and Shabestari, Behrouz
and Xing, Lei",
title="CNNs vs. Transformers: Performance and Robustness in Endoscopic Image Analysis",
booktitle="Applications of Medical Artificial Intelligence",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="21--31",
abstract="In endoscopy, imaging conditions are often challenging due to organ movement, user dependence, fluctuations in video quality and real-time processing, 
which pose requirements on the performance, robustness and complexity of computer-based analysis techniques. This paper poses the question whether Transformer-based 
architectures, which are capable to directly capture global contextual information, can handle the aforementioned endoscopic conditions and even outperform the 
established Convolutional Neural Networks (CNNs) for this task. To this end, we evaluate and compare clinically relevant performance and robustness of CNNs and 
Transformers for neoplasia detection in Barrett's esophagus. We have selected several top performing CNN and Transformers on endoscopic benchmarks, which we have 
trained and validated on a total of 10,208 images (2,079 patients), and tested on a total of 4,661 images (743 patients), divided over a high-quality test set 
and three different robustness test sets. Our results show that Transformers generally perform better on classification and segmentation for the high-quality challenging 
test set, and show on-par or increased robustness to various clinically relevant input data variations, while requiring comparable model complexity. 
This robustness against challenging video-related conditions and equipment variations over the hospitals is an essential trait for adoption in clinical practice. 
The code is made publicly available at: https://github.com/BONS-AI-VCA-AMC/Endoscopy-CNNs-vs-Transformers.",
isbn="978-3-031-47076-9"
}
```
and the following citation for the journal paper:
```bash
@article{KUSTERS2025103348,
title = {Will Transformers change gastrointestinal endoscopic image analysis? A comparative analysis between CNNs and Transformers, in terms of performance, robustness and generalization},
journal = {Medical Image Analysis},
volume = {99},
pages = {103348},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2024.103348},
url = {https://www.sciencedirect.com/science/article/pii/S1361841524002731},
author = {Carolus H.J. Kusters and Tim J.M. Jaspers and Tim G.W. Boers and Martijn R. Jong and Jelmer B. Jukema and Kiki N. Fockens and Albert J. {de Groof} and Jacques J. Bergman and Fons {van der Sommen} and Peter H.N. {De With}},
keywords = {Endoscopic image analysis, Convolutional neural networks, Transformers, Robustness, Generalization},
abstract = {Gastrointestinal endoscopic image analysis presents significant challenges, such as considerable variations in quality due to the challenging in-body imaging environment, the often-subtle nature of abnormalities with low interobserver agreement, and the need for real-time processing. These challenges pose strong requirements on the performance, generalization, robustness and complexity of deep learning-based techniques in such safety–critical applications. While Convolutional Neural Networks (CNNs) have been the go-to architecture for endoscopic image analysis, recent successes of the Transformer architecture in computer vision raise the possibility to update this conclusion. To this end, we evaluate and compare clinically relevant performance, generalization and robustness of state-of-the-art CNNs and Transformers for neoplasia detection in Barrett’s esophagus. We have trained and validated several top-performing CNNs and Transformers on a total of 10,208 images (2,079 patients), and tested on a total of 7,118 images (998 patients) across multiple test sets, including a high-quality test set, two internal and two external generalization test sets, and a robustness test set. Furthermore, to expand the scope of the study, we have conducted the performance and robustness comparisons for colonic polyp segmentation (Kvasir-SEG) and angiodysplasia detection (Giana). The results obtained for featured models across a wide range of training set sizes demonstrate that Transformers achieve comparable performance as CNNs on various applications, show comparable or slightly improved generalization capabilities and offer equally strong resilience and robustness against common image corruptions and perturbations. These findings confirm the viability of the Transformer architecture, particularly suited to the dynamic nature of endoscopic video analysis, characterized by fluctuating image quality, appearance and equipment configurations in transition from hospital to hospital. The code is made publicly available at: https://github.com/BONS-AI-VCA-AMC/Endoscopy-CNNs-vs-Transformers.}
}
```



# Detection of floating litter with image classification

This repository contains the code used for the following publication:
```bash
  Jia T, Vallendar AJ, de Vries R, Kapelan Z and Taormina R (2023) Advancing deep learning-based detection of floating litter using a novel open dataset. Front. Water 5:1298465. doi: 10.3389/frwa.2023.1298465
```

The aim of this study is to conduct a thorough evaluation of the performances of five deep learning architectures (ResNet50, InceptionV3, DenseNet121, MobileNetV2, and SqueezeNet) that detect floating macroplastic litter with multi-class image classification. The study also evaluates the benefits of multiple transfer learning strategies and data augmentation techniques on detection performance, and assess and improves the generalization ability of architectures considering unseen litter items and new device settings. 


## Dataset

"TU Delft - Green Village" (TUD-GV) dataset is a novel labelled
dataset of around ten thousand camera and phone images of floating macroplastic and other
litter items, collected from semi-controlled experiments in a drainage canal of the TU Delft Campus, the Netherlands. This dataset and further details can be found in:

```bash
  https://doi.org/10.5281/zenodo.7636124
```


## Requirements:
- Windows 10
- Python == 3.8.5
- Tensorflow==2.6.0
- Keras==2.6

```bash
  pip install -r requirements.txt
```

## Usage

-  `main_Model_training.ipynb` is the code for training deep learning models.
- `main_Evaluation.ipynb` is the code for evaluating model performnaces on test sets.
-  `main_Data_Augmentation.ipynb` is the code for generating images using data augmentation techniques.
-  `config.yaml` file is for defining parameters and other properties for the study.


## Authors

- [@Tianlong Jia](https://github.com/TianlongJia)
- [@Andre Jehan Vallendar](https://github.com/ajv95)
- [@Riccardo Taormina](https://github.com/rtaormina)

## Citation
If this repository helps your research or you use the aforementioned dataset for a publication, please cite the paper. Here is a BibTeX entry:

```BibTeX
@article{jia2023advancing,
  title={Advancing Deep Learning-based Detection of Floating Litter using a Novel Open Dataset},
  author={Jia, Tianlong and Vallendar, Andre Jehan and de Vries, Rinze and Kapelan, Zoran and Taormina, Riccardo},
  journal={Frontiers in Water},
  volume={5},
  pages={1298465},
  publisher={Frontiers},
  year={2023}
}
```
## Contact

➡️ Tianlong Jia ([T.Jia@tudelft.nl](mailto:T.Jia@tudelft.nl))

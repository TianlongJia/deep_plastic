
# Detection of floating macroplastic litter with image classification

The objective of this study is to develop five deep learning models (ResNet50, InceptionV3, DenseNet121, MobileNetV2, and SqueezeNet) that detect floating macroplastic litter with multi-class image classification. The study also evaluates the benefits of multiple transfer learning strategies and data augmentation techniques on detection performance, and assess and improves the generalization ability of architectures considering unseen litter items and new device settings.

If you happen to use this code for a publication, please cite the following paper which describes models and dataset:

```bash
  To do: update the citation of the paper 
```
## Dataset

"TU Delft - Green Village" (TUD-GV) dataset is a novel labelled
dataset of around ten thousand camera and phone images of floating macroplastic and other
litter items, collected from semi-controlled experiments in a drainage canal of the TU Delft
Campus, the Netherlands. 

```bash
  To do: update the Zenodo link
```


## Requirements:
- Windows 10 (It is only tested on Windows 10)
- Python == 3.8.5
- Tensorflow==2.6.0
- Keras==2.6

```bash
  pip install -r requirements.txt
```
## Usage

-  Edit the main_Model_training.ipynb file in the repository to train models using your desired deep learning architecture and training procedure.
-  Run the main_Evaluation.ipynb file to evaluate model performnaces on test sets.
-  Edit the main_Data_Augmentation.ipynb file to generate images using your desired data augmentation technique.
-  Edit the config.yaml file in the repository to define parameters and other properties for the project.
-  Use the IPython notebook for visualizing the results, unless you want to do otherwise.


## Authors

- [@Tianlong Jia](https://github.com/TianlongJia)
- [@Andre Jehan Vallendar](https://github.com/ajv95)
- [@Riccardo Taormina](https://github.com/rtaormina)

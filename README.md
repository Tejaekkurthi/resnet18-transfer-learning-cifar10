# From Frozen Features to Fine-Tuning

Fine-tuning a pretrained ResNet-18 vision backbone on CIFAR-10, comparing feature extraction against fine-tuning.

## Setup

Install the required packages:

pip install -r requirements.txt

The project uses PyTorch, torchvision, NumPy, and Matplotlib.

## Run

Open the following Jupyter Notebook:

resnet18_transfer_learning_cifar10.ipynb

Run the notebook cells in order.

The CIFAR-10 dataset is downloaded automatically using torchvision.

The experiment uses a random seed of 42 and the same training and validation split for both training runs.

For feature extraction, the pretrained ResNet-18 backbone is frozen and only the new classification head is trained.

For fine-tuning, the last residual block, layer4, is unfrozen and trained together with the classification head using discriminative learning rates.

The frozen backbone is kept in evaluation mode during feature extraction. If it were left in training mode, the BatchNorm running statistics would continue to update from the CIFAR-10 data even though the backbone weights were frozen. This could change the pretrained feature representation.

## Dataset

CIFAR-10 was used for this experiment.

The original CIFAR-10 training dataset contains 50,000 images across 10 classes.

A subset of 5,000 images was used:

Training images: 4,000  
Validation images: 1,000  
Number of classes: 10

The training transform uses:

RandomResizedCrop(224)  
RandomHorizontalFlip()  
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  
ImageNet normalization

The validation transform uses:

Resize(256)  
CenterCrop(224)  
ImageNet normalization

Random augmentation is used only during training. The validation pipeline uses a fixed view so that evaluation remains consistent.

## Results

| Stage | Val Accuracy | Train Time | Trainable Params |
|---|---:|---:|---:|
| Feature extraction | 0.769 | 552.73s | 5,130 |
| Fine-tuning | 0.743 | 811.54s | 8,398,858 |

## Conclusion

Feature extraction achieved the higher validation accuracy on this dataset.

The frozen ResNet-18 backbone already contained useful pretrained visual features, so training only the new classification head produced a validation accuracy of 76.9%.

Fine-tuning allowed the last residual block to update its more task-specific features for CIFAR-10. However, with the smaller dataset subset and three training epochs, this additional adaptation did not improve validation accuracy.

Fine-tuning also required more training time because many more parameters were trainable.

## Known limitations

Only 5,000 CIFAR-10 images were used instead of the complete training dataset to reduce training time.

Both approaches were trained for only three epochs. More training data, additional epochs, or further learning-rate tuning could produce different results.

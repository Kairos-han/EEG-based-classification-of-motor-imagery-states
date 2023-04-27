# EEG-Based Classification of Motor Imagery States
This repository provides an implementation of motor imagery electroencephalogram (EEG) signal classification based on convolutional neural networks (CNNs). The repository's code performs classification of EEG signals that correspond to imagined movement using a modified EEG-Inception neural network architecture.

## Dataset
The EEG dataset used in this project is provided in this repository.

## CNN Model
The CNN model used for classification in this project is based on the EEG-Inception neural network architecture. The code provides a modified version of the architecture that uses fewer parameters compared to the original model, and is optimized for EEG data classification.

Initially, a 1D-LeNet model was implemented to classify the EEG signals, but the results were suboptimal, achieving only 50% accuracy. Therefore, the modified EEG-Inception neural network model was implemented, which improved the classification accuracy substantially.

## Usage
The code for the project is contained within a Jupyter notebook file (.ipynb file extension). The file reads in the EEG dataset, preprocesses the data, builds the modified EEG-Inception network model, and trains the model on the data. The resulting model is then used to predict the motor imagery states of new EEG signals.

Users can modify the code parameters such as the number of training epochs or learning rate in order to customize the model.

## Dependencies
This project requires several Python packages to be installed, including tensorflow, numpy, matplotlib, sklearn, pandas, and scipy. Installation instructions can be found on their respective websites.

## Future Work
This project can be extended in several directions in the future. One potential direction is to use other machine learning models, such as Recurrent Neural Networks (RNNs), Attention-based Neural Networks or adversarial training with generative neural networks, and compare their performance with the modified EEG-Inception architecture. Another direction is to explore transfer learning approaches that can improve the accuracy of the model, especially when the size of the dataset is small.

## License
This project is licensed under the MIT License.

## References
Chollet, F. (2016). Xception: Deep Learning with Depthwise Separable Convolutions, arXiv preprint arXiv:1610.02357

Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. Journal of Neural Engineering, in press.

Zhang, X., Ma, L., & Wu, G. (2018). A Comparative Study on Convolutional Neural Network-based Classification of Motor Imagery EEG Signals, Journal of Neural Engineering

# EEG-Based Classification of Motor Imagery States
This repository provides an implementation of motor imagery electroencephalogram (EEG) signal classification based on convolutional neural networks (CNNs). The repository's code performs classification of EEG signals that correspond to imagined movement using a modified EEG-Inception neural network architecture.

This repository contains the code and datasets for the EEG classification project using CNN. The project was developed as part of the "Human-computer interaction technology" course at Huazhong University of Science and Technology, School of AI and Automation in 2022. 华中科技大学，人工智能与自动化学院，人机交互技术，课程设计

## Project Overview

The project focuses on exploring the application of CNNs in classifying EEG signals obtained from healthy subjects performing motor imagery tasks. The dataset consists of EEG recordings from 8 subjects, with two types of motor imagery tasks: right hand and both feet. The data is preprocessed and provided in .npz and .mat formats. 

The project includes the following sections:

1. Dataset Description: Provides an overview of the data collection process and details about the dataset structure.
2. Experimental Procedure: Describes the steps involved in loading the dataset, visualizing the data, and implementing the CNN models.
3. Model Evaluation: Presents the evaluation results of the 1D-LeNet model and the modified EEG-Inception Neural Network model.
4. Predicting the Test Set: Describes the process of predicting the test set using the saved CNN model.
5. Conclusion: Summarizes the findings of the study and discusses the challenges and future improvements.

## Repository Structure

The repository is organized as follows:

- `CNN_EEG.ipynb`: Jupyter Notebook file containing the code implementation.
- `datasets/`: Directory containing the dataset files in .npz and .mat formats.
- `results/`: Directory containing the results of the model evaluation and predictions.
- `README.md`: This file, providing an overview of the repository and the project.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/Kairos-han/EEG-based-classification-of-motor-imagery-states.git`
2. Open `CNN_EEG.ipynb` in Jupyter Notebook or any compatible environment.
3. Ensure that the required dependencies are installed (mentioned in the notebook).
4. Run the code cells in the notebook to reproduce the experiments and analyze the results.
5. Explore the `datasets/` and `results/` directories for the dataset files and generated results.

## Dependencies

The project code has the following dependencies:

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow

## References

[1] Zhang, C., Kim, Y. K., & Eskandarian, A. (2021). EEG-inception: an accurate and robust end-to-end neural network for EEG-based motor imagery classification. Journal of Neural Engineering, 18(4), 046014.
[2] Xu, B., Zhang, L., Song, A., Wu, C., Li, W., Zhang, D., ... & Zeng, H. (2018). Wavelet transform time-frequency image and convolutional network-based motor imagery EEG classification. Ieee Access, 7, 6084-6093.
[3] Tabar, Y. R., & Halici, U. (2016). A novel deep learning approach for classification of EEG motor imagery signals. Journal of neural engineering, 14(1), 016003.
[4] Aggarwal, S., & Chugh, N. (2019). Signal processing techniques for motor imagery brain computer interface: A review. Array, 1, 100003.


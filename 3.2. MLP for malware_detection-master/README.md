# Multilayer Perceptron for Malware Detection

This is a 2-student group project for CS5242 Neural Networks and Deep Learning taken under the Master of Science in Business Analytics programme in NUS. The project centers on using using deep learning models to detect malware from binary files extracted from PE header of exe files The project report is available (https://drive.google.com/file/d/1kvfmgn1btlGX5t0_XnJhdfvgrVzdgbjt/view?usp=sharing) to explain the approach and results obtained. The codes were predominantly written and compiled by me while my groupmate and I shared the credit for formulating the approach and running the models. As this is a course on neural networks, multilayer perceptrons or MLP (deep, fully connected neural networks) were the main models used. Other machine learning algorithms were not considered other than the use of support vector machine for ensembling the MLP models. Data used were not included as distribution was not allowed.

data.py: read train.csv, train_label.csv and test.csv for data pre-processing(feature selection, normalization etc)

models.py: specifications for 10 neural network model architectures

train.py: reads from data.py and models.py to train all 10 neural network models

predict.py: reads from data.py and models.py to predict from neural network models and perform stacking using SVM. Outputs predictions file for submission

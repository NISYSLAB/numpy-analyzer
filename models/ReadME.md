# Classification of Electrogastrogram (EGG) Signals into functional states.
Electrogastrography (EGG) is an examination method for investigating the myoelectrical activity of a stomach. EGG is widely used in research and diagnosis of GI diseases such as nausea, vomiting, functional dyspepsia, gastroparesis, motion sickness, etc. 
In our study, we demonstrate the usage of machine learning algorithms to predict the functional states of the stomach. Our dataset consists of two states, fasting and postprandial. We utilize feature engineering techniques to extract time domain features, frequency domain features, and fractal features. We then use the extracted features to classify the signals using SVM and Random Forrest algorithms. We obtained 89% test accuracy with SVM, 93% test accuracy with Random Forest, and 95% test accuracy with Neural Networks.

For detailed explaination of the study refer to the full report [here](Classification_Report.pdf)

#### Project File Structure

- **EGG_ML_RF.ipynb:** Contains the Random Forrest (RF) model, the hyperparameters experiment for the model, and the model's final results confusion matrix.
- **EGG_ML_SVM.ipynb:** Contains the Support Vector Machine (SVM) model, the hyperparameters experiment for the model, and the model's final results confusion matrix.
- **EGG_DL_NN.ipynb:** Contains the Dense Neural Network structure, training and testing experiments, and the model's final results confusion matrix.
- **EGG_DL_MODEL.h5:** Contains the model generated from the EGG_DL_NN file
- **Features_data_df.csv:** Contains the features dataframe generated from the dataset and used in training and testing the models. This file was generated from **load-data-build-features-df.ipynb** jupyter notebook which exists in the package-functions folder.

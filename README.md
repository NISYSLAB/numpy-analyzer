![Google Summer of Code Logo](data/GSoC.png)
![Emory University - BMI Department Logo](data/emory-bmi.jpg)

# Python based tool for viewing and basic analysis of files with Numpy format


This project was developed as part of the Google Summer of Code (GSoC) 2022 program. 

> Contributor : Nada Ahmed Elmasry
 
> Mentor : Mahmoud Zeydabadinezhad

> Organization : Department of Biomedical Informatics (BMI), Emory University School of Medicine

## Introduction
The project contains an open-source Python package for loading, analysis, and visualization of electrophysiological data. The package will enable users to perform various types of analysis and visualization of data in an easy way. This will free users to focus on the results and interpretation of the analysis and visualizations.


In Addition to the package, the project contains a research study where we managed to classify Electrogastrogram (EGG) signals into their functional states using Machine learning and Deep learning models.



## Contributions

### EGG signal Classification ML and DL models


The aim of this study was to classify the EGG signals into two functional states: fasting and postprandial. This was achieved using Support Vector Machines and Random Forrest ML algorithms, and through a Dense Neural Networks (DNN) model.


For Models implementation details and results refer to the study code [here](models/ReadME.md)


For detailed explaination of the study and the classification pipeline please refer to the full report written [here](models/Classification_Report.pdf)




### Python based tool for analysis and visualization of electrophysiological signals


The viewer is a python based tool for efficient analysis and visualization of electrophsyiological signals. The viewer can be used as imported package inside the user’s code where the user can call individual functions or from the GUI where the user can choose parameters for the analysis and visualization from an easy to use interface.


To see package source code and download GUI refer to the package [ReadME](package-functions/ReadME.md)


For detailed explaination of the viewer's structure and functions please refer to the the full report written [here](package-functions/Viewer-Report.pdf)




# Genetic optimization of deep CNN based transfer learning for COVID-19 identification from radiography chest X-Rays
** **
***This repository includes a meta-heuristics approach to optimize transfer learning based deep cnn model (primarily vgg16) for low uncertainty in covid-19 identification using x-ray images.***
** **

**A. Environment**
+ [Google Colaboratory](https://colab.research.google.com "Google Colab"): This project use Google Colaboratory for compute purpose.
+ RAM: 12 GB, Disk Space: 80 GB
+ Compute Engine: GPU
+ Language: Python

**B. Dataset**
+ [Covid-19 Radiography Dataset](https://drive.google.com/drive/folders/1i_kQHjdOYFOyaOsI3mFdG8Deabi4dvOt "Covid-19 X-Ray Images")
  - M. E. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M. A. Kadir, Z. B. Mahbub, K. R. Islam, M. S. Khan, A. Iqbal, N. Al Emadi, et al., Can ai help in screening viral and covid-19 pneumonia?, IEEE Access 8 (2020) 132665–132676.
  - T. Rahman, A. Khandakar, Y. Qiblawey, A. Tahir, S. Kiranyaz, S. B. A.395 Kashem, M. T. Islam, S. Al Maadeed, S. M. Zughaier, M. S. Khan, et al., Exploring the effect of image enhancement techniques on covid-19 detection using chest x-ray images, Computers in biology and medicine 132 (2021) 104319.
+ **Sample Data:** Following is the sample of dataset with class label. <br> ![Sample Dataset](/Pictures/Sample_Data.png "Sample Dataset")

**C. Experimental Data**
+ [Pickle Data](https://drive.google.com/drive/folders/1gnx-tpOwSDpnYJFmMyLau12pjYqekb8X "Experimental Data"): All experimental data of this project work are stored in [Pickle Data](https://drive.google.com/drive/folders/1gnx-tpOwSDpnYJFmMyLau12pjYqekb8X "Experimental Data").
+ [Data](https://drive.google.com/drive/folders/17cspYJS7XeGflOzu5_g2rEpuLtr8dEF_ "Training Histories"): All training histories are stored in this [link](https://drive.google.com/drive/folders/17cspYJS7XeGflOzu5_g2rEpuLtr8dEF_? "Training Histories").
+ [Models](https://drive.google.com/drive/folders/1wmcpabmqLIaDCWFhYOdzjUqDBCTCLsfp "Best Optimized Models"): All best models data found during optimization and exploitation are stored [here](https://drive.google.com/drive/folders/1wmcpabmqLIaDCWFhYOdzjUqDBCTCLsfp "Best Optimized Models").
+ [TL-Models](https://drive.google.com/drive/folders/1uKNctQweu3tPD74sU7MY0XcKobA-2MBv "Best Transfer Learning Models"): All best models data found during training of transfer learning models are stored [here](https://drive.google.com/drive/folders/1uKNctQweu3tPD74sU7MY0XcKobA-2MBv "Best Transfer Learning Models").

**D. Codes**
+ **Model: **
  - `VGG16`: VGG16 model primirily used for this project. Following figure show the structure of this CNN model. <br> ![VGG16](/Pictures/VGG16_CNN.png "VGG16")
+ ***Transfer_Learning_with_Optimization_of_Covid_19_Image_Classification.[ipynb](https://github.com/jahid-jabed/mh_opt_tl_covid19/blob/main/Codes/IPYNB/Transfer_Learning_with_Optimization_of_Covid_19_Image_Classification.ipynb)/[py](https://github.com/jahid-jabed/mh_opt_tl_covid19/blob/main/Codes/PY/transfer_learning_with_optimization_of_covid_19_image_classification.py):*** Codes on optimization process on transfer learning for covid-19 identification.
  - `collect_Data()`
    * This function used to collect data based on the classes (folders, each folder represent a class in dataset folder) from given drive path.
  - Pickle
    * Pickle is used for store and retrive data from given drive path.
  - `train_test_split(...)
    * Used for data spliting for cross validation during training.
  - `display_multiple_img(...)`
    * Used for display some random data images with thier classes.
  - `train(...), compile(...), fit(...), ...` etc
    * Used for training the model and performance measures.
  - `initial_individuals(model)`
    * Used for initialization of unique individual for the initial population.
  - `mutate(individual, model)`
    * Used for mutation of `individual` with a mutation factor and provide unique gene.
  - `crossover(individuals, model)`
    * Used for crossover operation on a population (`individuals`) and provide unique `individual` in population using uniform crossover.
  - `evolve(individuals, fitness, model)`
    * Used for evolution of `individuals` based on `fitness` for next generation. In this work evolution is done upto 30 generations.
  - Performance Measures
    * After optimization of models performance measures for best 3 models are conduct based on accuracies, confusion matrix, etc.
    
+ ***Exploitation_on_Transfer Learning_with_Optimization_Covid_19_Image_Classification.[ipynb](https://github.com/jahid-jabed/mh_opt_tl_covid19/blob/main/Codes/IPYNB/Exploitation_on_Transfer%20Learning_with_Optimization_Covid_19_Image_Classification.ipynb)/[py](https://github.com/jahid-jabed/mh_opt_tl_covid19/blob/main/Codes/PY/exploitation_on_transfer_learning_with_optimization_covid_19_image_classification.py):*** Codes on further exploitation process on optimized transfer learning models found for covid-19 identification.
  - This python file includes codes for training of optimized transfer learning models found for further 100 epochs.
  
+ ***Transfer Learning_without_Optimization_Covid_19_Image_Classification.[ipynb](https://github.com/jahid-jabed/mh_opt_tl_covid19/blob/main/Codes/IPYNB/Transfer%20Learning_without_Optimization_Covid_19_Image_Classification.ipynb)/[py](https://github.com/jahid-jabed/mh_opt_tl_covid19/blob/main/Codes/PY/transfer_learning_without_optimization_covid_19_image_classification.py):*** Codes on straight forward transfer learning training process for covid-19 identification.
  - This python file includes codes for training of straight forward transfer learning models without optimization for 300 epochs and saved best models found.
  - Model construction is done randomly based on fair coin toss.

**E. Results and Performance**
  - **Results:** Performance of the Transfer Learning Models with and without Genetic Optimization. <br> ![Results](/Pictures/Results.PNG "Results")

**F. Dependencies**
  + The dataset importation from Google Drive and saving model data into Google Drive required drive permission.
  

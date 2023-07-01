# NeuroTechX-Alzheimers-Identification
This project was created for the NeuroTechX summer 2023 competition. It analyzes several ML/DL methods for Alzheimer's and fronto-temporal dementia detection. Tools such as ICA, PCA, PLSDA, and CWT were used to prepare data for models including SVM, KNN, random forests, gradient boosted decision trees, MLP, and CNN. 

The DL_GBDT file contains implementations for the MLP, gradient boosted decision tree, and CNN models in that order. To replicate results, you will need to download the data and change the filepaths to the correct location. You will then be able to run the cells in order to process the data and retrieve the results. 
NOTE: ALL CELLS UNDER "Create images from cwt data and store in Drive if necessary for space" ARE COMPUTATIONALLY EXPENSIVE. This data processing is only required for the CNN model and can be skipped for all other purposes. 

Further details about specifics of data setup and other models used in the "Classical" file can be found below: 
After downloading all of the data, change the variable folder_name at the top of the code to the path of the folder containing all of the data. Be sure to not include the “/” (or “\”) at the end of the path.

Below the declaration of folder_name is code for visualizing the preliminary data for any patient. Change the variable subject to the patient number to visualize the data for that patient. There are 88 patients in total, so valid values for subject are integers from 1 to 88, inclusive. The function visualize_data() is then called, plotting both the raw EEG data and ICA components of the patient.

Functions epoch_signal() and get_bandpowers() are helper functions for preprocessing the data and extracting band powers as features.

Variable subj_types is a 1D array containing the class labels of each of the 88 patients in order. These are imported from the participants.tsv file found in the link above.

Function get_subject_features() is a helper function that extracts, preprocesses, and converts the data of a given patient into features. It returns a 2D dataframe where the columns are features (8 for each of the 19 channels, 152 in total) and the rows are epochs (each patient’s EEG data is divided into 4-second epochs). get_subject_features() is called by extract_all_features(), which calls it for each of the 88 patients.

Function split_data() randomly splits the patients into training and testing groups in a stratified fashion. 70% of each diagnosis group (Alzheimer’s, FTD, control) goes into the training set, while 30% goes into the testing set. Each set is concatenated into a single dataframe with columns as features and rows as epochs; each epoch will be treated as a single sample. The data is scaled by features to a standard range, resulting in scaled_X_train and scaled_X_test. Lists flat_y_train and flat_y_test are the class labels of the samples in the training and testing sets, respectively.

Functions get_pca_data() and get_plsda_data() transform the data via principal component analysis (PCA) and partial least squares-discriminant analysis (PLS-DA), respectively, in hopes of increasing the accuracy of the learning models. The transformed data did not perform significantly better than the untransformed data, so these functions are unused and not included in the video.

We used three classical machine learning methods for classification: random forest, support vector machine (SVM), and K-nearest neighbors (KNN). 

Random forest uses an ensemble of decision trees, where each decision tree independently takes a random subset of features to train on and classify a sample into a group, and a final decision is made through majority voting. We used 100 decision trees and each tree takes the square root of the total number of features.

In SVM, samples are plotted into an N-dimensional space where N = the number of features and we find the optimal hyperplane that best separates datapoints of different classes using a linear function.

In KNN, a decision on the class of the testing sample is made based on the classifications of the k-nearest training samples to the testing sample. We chose to use the three nearest neighbors.

Each learning method has its own function (run_rf(), run_svm(), run_knn()) which takes in the training and testing data and labels and output the predicted labels of each sample and the accuracy the the model. These models are run together for any n times for n-fold validation. A confusion matrix is then generated for each model. A bar graph comparing the accuracy for each model is also generated.

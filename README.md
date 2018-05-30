# Selective-Learning
A two-step learning method that use probe learning firstly to find and remove outliers, and then use new training set to train and predict

Selective learning is a method designed to improve accuracy of predictions, test models including Gradient Boost, Random Forest and Support Vector Machine.

This method is based on the fact that there are always some extreme data that are not as normal as most ones. 
These data need to be treated separately, or they will have a bad influence on the whole dataset for prediction.
Therefore, the core is to partition the training set into train_train and train_test firstly, and then use probe learning to pro-process the data and find those with worst accuracy.
After removing those "bad guys" and their "neighbours", which are similar to them, conduct machine learning on the new data set.


This file contains several methods, which are more extensively described in their respective files.
**The run.py file is a script than runs the model and reproduces the best results we had with the model in the results folder.** 
(It will replace what already is in that folder).

The results will be produced the following way :
- It will split the training dataset into 4 datasets according to the JET_split column of our data
- It will perform a 4-fold cross validation for each of the splits : iterate over the degrees 9,10,11,12,13,14 and lambdas = np.logspace(-3,3,20)
- After finding the best degree and lambda for each split, it trains all the data with these parameters.
- Then we do the prediction on the testing data.


Call run.py and some informations about the progress of the simulation will be printed into the terminal. The outcome of the prediction will be exported to the results folder, under the name output_sanitized_normalization_test_split_data_9_10_11_12_13__14_lambda_3_3_20.csv

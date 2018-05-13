# Model training and prediction process
1. Iteratively training 8 model based on the different transformations.  
2. While creating the cleaned and appropriate data (i.e. Vector) for training, data is partitioned them by row.  
3. As the Random Forest Model partitions the feature while training, at that time the data is partitioned by features i.e. Column.  
4. While testing, the data on the models is run iteratively and predicting the final value through Polling.  

# Final Results:  
- Algorithm: Random Forest Classifier  
- Parameters:  
1. Number of trees = 20  
2. Depth = 5  
3. Training data size = 30 * 8 ~ 240 GB  
4. Accuracy = 99.74%  
- Time to Train models = 100 minutes on 18 m4.large  
- Time to run the Test data = 7 minutes on 15 m4.large  

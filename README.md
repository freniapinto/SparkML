Model training and prediction process
1. Iteratively training 8 model based on the different transformations.  
2. While creating the cleaned and appropriate data (i.e. Vector) for training, we are partitioning them by row.  
3. As the Random Forest Model partitions the feature while training, at that time the data is partitioned by features i.e. Column.  
4. While testing, we are running the data on the models iteratively and predicting the final value through Polling.  

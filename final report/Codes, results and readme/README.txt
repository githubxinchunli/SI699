# The sample_dataset has sampled 300 records for users.csv and 30000 for tweets.csv (as the account level may need more tweets to run successfully)
# You may put the sample_dataset folder in the same directory with the .py file to run the code. For example:
 - sample_dataset
 	- cresci-2017.csv
 		- datasets_full.csv
 			- fake_followers.csv
 				- tweets.csv
 				- users.csv
 	- russian-troll-tweets
 - profile_feature_boosting_model.py
 - profile_feature_lr_model.py
 - profile_feature_rf_model.py
 - profile_feature_svm_model.py
 - text_feature_boosting_model.py
 - text_feature_lr_model.py
 - text_feature_rf_model.py
 - text_feature_svm_model.py

# The profile_feature_ModelName_model.py files are corresponding to the basic feature models.
# The text_feature_ModelName_model.py files are corresponding to the text feature models.
# The .csv results are just examples, please check the output files for the true results.



# The csv files of performance results are included in the according folders as well. There might be more than one csv file that documents the results, which are basically a raw version and a summary version. They are just of different formats to assist with visualization. There are visualization ipynb files included in the text-based and simple-numeric features based methods, which would require import from according results csv files. To be able to run the visualization files, you would need to include the corresponding csv file in the same folder. 

# One or two visualization in the report is done in Excel, and the rest is done in Jupyter  Notebook. 

# Please note that in the tweets level folders under text based and under Simple numeric features based folders, there are both in-domain and cross-domain ipynb files. These are the basecode we ran multiple times, and results are documented manually. So each single file is not the complete basecode of all the results. They are re-used again and again, with different input (change “b”, “c” or only “b” in the code). But otherwise, all the needed code is included. Same thing with the Baseline_models.ipynb file as well. This basecode has been run a lot of times with different input.

# And in order for the code to run properly with the smaller sampled dataset, we adjusted the sampling number in the files accordingly. (For example, in the text_tweet_cross_domain.ipynb, the sampling number included is 1000 for a1, 1000 for a2, 1000 for b1, 1000 for c1 and 100 for b2. Our original sampling number is 10000 for a1, 10000 for a2, 10000 for b1, 10000 for c1 and 1000/2000/3000/4000 for b2. You can see the according sections in our report for more detailed information about how we do sampling) Just FYI. 


Thank you for patiently reading through this document!! Please feel free to let us know anytime if you have any question, or for some reason, the code cannot run properly.



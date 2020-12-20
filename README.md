# 702CW

•	The group member details (if applicable)
•	Their project abstract or very short description 
•	Description about structure of code (code and data file locations) 
•	How to run the code ( main file name etc) 
•	Any output of the code (especially in the case of any graphs/plots etc, if they have)
•	Location/link to the data 
•	Also mention the github account in their reports so that we can quickly browse the correct submission.

Group members:
Kasper Groes Ludvigsen, kasper.ludvigsen@city.ac.uk
Hugh Adams, hugh.adams@city.ac.uk

## Task 2 specific files:
main.py
      This is the file from which the neural network can be called. Open the file in an IDE an run the whole file at once
neural_network.py
      This is the file in which the neural network class is implemented. The class is caled from main.py
unittest  
      A folder that contains the file unittest_neural_network.py containing unit tests related to the development of the task 2 model. Don't run the code in this file as some of         the unit tests have not been updated to reflect refactoring of code.
load_data.py
      This file should not be run. It contains a helper function used to generate the MNIST dataset for task 2. 
      
## Task 3 specific files:
Task3ffn_early_stopping.py
      This file should be run. It contains the code that is used to train models that are comparable to the model developed in task 2. Open the file in an IDE an run the whole           file at once
      
task3_cnn_early_stopping.py
      This should be run. The file contains the code that is used to train the CNN model reported in task 3. Open the file in an IDE an run the whole file at once

## Task 4 specific files:
task4_main.py
      This files should be run. The file is the main file for the models presented in task 4. Open the file in an IDE an run the whole file at once. 

task4_data.py
      This file should not be run. It contains methods developed for feature engineering and data pre processing related to task 4. The functions in the file are called from           task4_main.py

t4unittest
      This file should not be run. It contains unit tests related to task 4. Some of the unit tests have not been updated after refactoring of methods. 

df_all_data_w_desc_2020-12-14.csv
      This file contains the raw data used for task four.

## Other files:
utils.py
      This file should not be run. It contains a range of helper functions used across tasks.
pytorch_dataset.py
      This file should not be run. It contains a method used for converting the data in task 4 to Pytorch compatible format
RecordModelPerformance.py
      This file should not be run. It contains a method used for recording model configuration and performance during experiments
      
Various other files including png, xlsx etc should not be considered as the most important plots etc. are found in the report. 

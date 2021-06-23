
****************************************
All submitted files and their content:
	1. Python code: All the Python code that we developed for this project. 
	2. Dataset: The dataset we collected, as well as a file "source_of_images.txt" detailing the source of each image. As the full dataset is too large to be in the submission, go to below link to download our dataset: https://drive.google.com/drive/folders/1XmEWv2rwU8I_m09c6KwFqylxyHu4GKw8?usp=sharing
	3. README: A readme.txt that lists all submitted files, instructions to run our code.
	4. Report: The project report, as detailed in the project description, in PDF format.
****************************************
How to run our code:
	!!!Attention!!!: As the dataset is too large to be submitted, we didn't put the full dataset in the code part. If you need to run our code, please first go to above mentioned link, download the full dataset, then replace the "train" folder and "test" folder with the downloaded full dataset, then you can run our code.

	run: python main.py

	By running "python main.py", our code will apply 10-fold cross validation on 80% of the training data. After finishing 10-fold cross validation, it will run the trained CNN on the testing data (20%). The evaluation and result will be printed in console, as shown in the report, including tables of results showing the accuracy, precision, recall and F1-measure of each iteration, the avearge of measurements of all 10 iteration, as well as confusion matrices in both the console (plain version) and the pop up window (graphical version).
****************************************

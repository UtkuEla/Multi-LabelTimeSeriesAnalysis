# ProjectArbeit
Repo for Summer23 Project Arbeit

test_data and train_data folders contains the datasets I have been using.

main.py is the main script for combining all the other code and training.

However, I am using google colab for training and testing ideas so the main.py script is not up-to-date.

At the end, main.py will be the final script.

readme will be updated.

# ideas to try:

300 - 150 - 50 sampling rates with a 10% overlap

K-fold-Cross-Validation instead of train-test split:

in K-fold-Cross-Validation, you need to split your dataset into several folds, then you train your model on all folds except one and test model on remaining fold. You need to repeat this steps until you tested your model on each of the folds and your final metrics will be average of scores obtained in every fold. This allows you to prevent overfitting, and evaluate model performance in a more robust way than simple train-test.

Examining more metrics since 98% accuracy is good but might be hiding the big picture!

Take away the NoFault and Outliers class and try to model again!

Multi-Label classification from a single-label dataset is the keyword!

Z-normalization for drift dataset(will be applied to all)! Maybe it can solve the dominance issue.




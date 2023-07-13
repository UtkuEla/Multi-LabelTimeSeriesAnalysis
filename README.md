# ProjectArbeit
Repo for Summer23 Project Arbeit

test_data and train_data folders contains the datasets I have been using.

main.py is the main script for combining all the other code and training.

However, I am using google colab for training and testing ideas so the main.py script is not up-to-date.

At the end, main.py will be the final script.

readme will be updated.

# ideas to try:

## 300 - 150 - 50 sampling rates with a 10% overlap

## K-fold-Cross-Validation instead of train-test split:

in K-fold-Cross-Validation, you need to split your dataset into several folds, then you train your model on all folds except one and test model on remaining fold. You need to repeat this steps until you tested your model on each of the folds and your final metrics will be average of scores obtained in every fold. This allows you to prevent overfitting, and evaluate model performance in a more robust way than simple train-test.

My remarks on k-fold-cv: I think it is not a very good method for time-series analysis since it breaks the sequential nature. However, in my case, the sequential nature of the data is not important, since I am taking data snippets and my task is not forecasting, but classification. As a result, k-fold-cv is logically applicable to my task.

## Examining more metrics since 98% accuracy is good but might be hiding the big picture!

I will add precision, recall, and hamming loss in addition to accuracy. In the end, my most crucial metric is the multilabel test dataset!!

## Take away the NoFault and Outliers class and try to model again!

They might be disturbing the homogenous nature of the dataset! After I had a successful model, I will make a data augmentation for Outliers and add it back! For the NoFault, it seems it might not be necessary to have it.

## Multi-Label classification from a single-label dataset is the keyword!

Very less example (almost none), existing ones are NLP and not quite applicable to my scenario.

## Z-normalization for drift dataset(will be applied to all)! Maybe it can solve the dominance issue.




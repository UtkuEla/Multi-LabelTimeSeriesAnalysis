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
These are the results of 30 pieces of training in K-fold-cross-validation:
Accuracy: 0.854, Max: 0.98, Min:0.38 
I think this suggests that there is an imbalance in the dataset.

[0.8128654970760234,
 0.97953216374269,
 0.8421052631578947,
 0.7017543859649122,
 0.38011695906432746,
 0.956140350877193,
 0.7807017543859649,
 0.8567251461988304,
 0.9649122807017544,
 0.9706744868035191,
 0.9736842105263158,
 0.9649122807017544,
 0.9532163742690059,
 0.6812865497076024,
 0.9678362573099415,
 0.827485380116959,
 0.97953216374269,
 0.9532163742690059,
 0.9502923976608187,
 0.4398826979472141,
 0.4619883040935672,
 0.9473684210526315,
 0.956140350877193,
 0.9532163742690059,
 0.9619883040935673,
 0.9385964912280702,
 0.956140350877193,
 0.8216374269005848,
 0.9298245614035088,
 0.7419354838709677]

 After testing without outliers and NoFault these are the results for 10 training:
 Accuracy: 0.990 (0.008) Max: 1.0, Min: 0.97

 [0.9963503649635036,
 0.9890510948905109,
 1.0,
 0.9744525547445255,
 0.9963503649635036,
 0.9817518248175182,
 0.9963369963369964,
 0.9853479853479854,
 0.9963369963369964,
 0.9853479853479854]

## Multi-Label classification from a single-label dataset is the keyword!

Very less examples (almost none), existing ones are NLP and not quite applicable to my scenario.

## Z-normalization for drift dataset(will be applied to all)! Maybe it can solve the dominance issue.




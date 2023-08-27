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




For multilabel classification tasks, where each sample can belong to multiple classes simultaneously, it is essential to use evaluation metrics that are appropriate for this scenario. Accuracy is not suitable for multilabel classification because it is designed for multiclass tasks where each sample belongs to only one class.

Instead, you should consider using the following evaluation metrics for multilabel classification:

Hamming Loss: Hamming loss measures the fraction of labels that are incorrectly predicted. It calculates the average fraction of incorrect labels for all samples. Lower values are better.

F1 Score (Micro and Macro): F1 score combines precision and recall and is a good metric for imbalanced multilabel datasets. It is available in two variants: micro-average (aggregating all true positives, false positives, and false negatives across all classes) and macro-average (calculating the metric per class and then averaging). F1 score ranges between 0 and 1, where 1 is the best possible score.

Jaccard Similarity Score (Jaccard Index): Jaccard similarity score measures the similarity between two sets, and in multilabel classification, it computes the average similarity between the true and predicted label sets for all samples. It ranges between 0 and 1, where 1 indicates perfect similarity.

Subset Accuracy (Exact Match Ratio): Subset accuracy measures the proportion of samples where the predicted labels exactly match the true labels. It's also known as the exact match ratio.

Precision, Recall, and Specificity: These are class-wise metrics that provide insights into the performance of the model for each individual label.



#====================================

18/18 [==============================] - 2s 35ms/step
[[1.78429937e-10 1.00000000e+00 5.38355449e-09 1.22834706e-15]
 [1.59950375e-09 3.97599651e-05 9.84706581e-01 5.45651512e-03]
 [9.93147075e-01 7.07266061e-03 1.17023745e-04 6.10383211e-07]
 ...
 [3.01881603e-10 2.45849387e-05 9.97476041e-01 8.67664232e-04]
 [5.45628147e-08 7.64076030e-05 5.94609261e-01 2.40660235e-01]
 [8.29223252e-04 6.75946940e-05 1.44438909e-05 9.99989033e-01]]
[[0 1 0 0]
 [0 0 1 0]
 [1 0 0 0]
 ...
 [0 0 1 0]
 [0 0 1 0]
 [0 0 0 1]]
>0.951
Hamming Loss: 0.02281021897810219
F1 Score (Micro): 0.9545454545454546
F1 Score (Macro): 0.955954567605053
Jaccard Similarity Score: 0.9543795620437956
Subset Accuracy: 0.9507299270072993
Confusion Matrix:
[[144   0   0   0]
 [  0 122   0   0]
 [  0   0 118  23]
 [  0   0   0 141]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       144
           1       1.00      1.00      1.00       122
           2       1.00      0.84      0.91       141
           3       0.86      1.00      0.92       141

    accuracy                           0.96       548
   macro avg       0.96      0.96      0.96       548
weighted avg       0.96      0.96      0.96       548

Model saved as /content/drive/MyDrive/ProjectArbeit/models/Sigmoid_long_31-07-15-48.h5


43/43 [==============================] - 3s 21ms/step
[[1.0000000e+00 1.0000000e+00 6.3309053e-16 1.5050742e-11]
 [1.0000000e+00 7.4310197e-11 1.0000000e+00 3.7960433e-08]
 [9.9987292e-01 4.6714609e-05 2.1895221e-06 1.8468909e-09]
 ...
 [2.5474102e-12 1.0000000e+00 2.9695440e-17 5.5015133e-18]
 [2.7640685e-13 1.0000000e+00 3.1075188e-20 3.3793201e-22]
 [6.6645723e-13 1.0000000e+00 1.8715591e-19 7.3475785e-20]]
[[1 1 0 0]
 [1 0 1 0]
 [1 0 0 0]
 ...
 [0 1 0 0]
 [0 1 0 0]
 [0 1 0 0]]
>0.615
Hamming Loss: 0.13249269005847952
F1 Score (Micro): 0.8394952402036749
F1 Score (Macro): 0.8413037137734202
Jaccard Similarity Score: 0.7984892787524366
Subset Accuracy: 0.6147660818713451
Confusion Matrix:
[[441  62  52   8]
 [ 36 368   1   0]
 [ 61   0 203  10]
 [  0   0   0 126]]
Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.78      0.80       563
           1       0.86      0.91      0.88       405
           2       0.79      0.74      0.77       274
           3       0.88      1.00      0.93       126

    accuracy                           0.83      1368
   macro avg       0.84      0.86      0.85      1368
weighted avg       0.83      0.83      0.83      1368


==========================================================

214/214 [==============================] - 3s 15ms/step
[[9.9988389e-01 4.2259380e-05 2.5657196e-06 1.8803636e-09]
 [9.9987388e-01 4.4921650e-05 2.0688547e-06 1.9659039e-09]
 [9.9987113e-01 4.6077319e-05 1.9341608e-06 1.9798569e-09]
 ...
 [3.7210457e-06 4.3367798e-16 1.0000000e+00 1.0000000e+00]
 [1.8997797e-13 3.2094588e-17 1.0000000e+00 1.0000000e+00]
 [1.6740527e-02 8.9500484e-04 9.9989617e-01 9.9475044e-01]]
[[1 0 0 0]
 [1 0 0 0]
 [1 0 0 0]
 ...
 [0 0 1 1]
 [0 0 1 1]
 [0 0 1 1]]
>0.858
Hamming Loss: 0.042909356725146196
F1 Score (Micro): 0.9477246415531214
F1 Score (Macro): 0.9492590088553188
Jaccard Similarity Score: 0.9269249512670564
Subset Accuracy: 0.8584795321637427
Confusion Matrix:
[[2614   62   52    8]
 [  37 2007    7    1]
 [  61    0 1297   10]
 [   0    0    0  684]]
Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.96      0.96      2736
           1       0.97      0.98      0.97      2052
           2       0.96      0.95      0.95      1368
           3       0.97      1.00      0.99       684

    accuracy                           0.97      6840
   macro avg       0.97      0.97      0.97      6840
weighted avg       0.97      0.97      0.97      6840

Model saved as /content/drive/MyDrive/ProjectArbeit/models/Sigmoid_long_total96_31-07-17-11.h5
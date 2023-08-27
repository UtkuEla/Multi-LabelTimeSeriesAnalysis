from testdata import *
from traindata import *

sampleSize = 300
overlapRatio = 10
path_train = "df_f_feather"
path_test = "test_df"

train = trainData()
#test = testData()

train_data_values, train_labels = train.prepareTrainData(sampleSize, overlapRatio, path_train)
#test_data_values, test_labels = test.prepareTestData(sampleSize, overlapRatio, path_test)

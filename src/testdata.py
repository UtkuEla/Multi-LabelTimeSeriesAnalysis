import pandas as pd
import numpy as np
import os

class testData():

    def __init__(self):

        self.current_folder_path = os.getcwd()
        self.parent_folder_path = os.path.dirname(self.current_folder_path)
        self.data_save_path = os.path.join(self.current_folder_path, 'test_data')


    def addDataPathtoCurrent(self,path):
        return os.path.join(self.current_folder_path,path)
    
    def addDatatoPath(self,path):
        return os.path.join(self.data_save_path,path)
    
    def readData(self,string):

        df = pd.read_pickle(string)

        if "time" in df.columns :
            df = df.drop(columns = {'time'})

        self.df = df.drop(index=0)

    def splitData(self):
        df = self.df
        df['label'] = df['label'].str.replace(' ', '')

        # Split the dataset based on labels
        labels = ['Bias+Drift', 'Bias+Gain', 'Bias+Precisiondegradation', 'Bias',
       'Bias+Outliers', 'Drift+Gain', 'Drift+Precisiondegradation',
       'Drift', 'Drift+Outliers', 'Gain+Precisiondegradation', 'Gain',
       'Gain+Outliers', 'Precisiondegradation',
       'Precisiondegradation+Outliers']

        self.datasets = {}
        self.dataframes =[]

        for label in labels:
            
            self.datasets[label] = df[df['label'] == label]

            globals()[f'df_{label}'] = self.datasets[label]
            self.dataframes.append(f'df_{label}')
            #datasets[label].to_csv(self.data_save_path + "/" +f'df_{label}.csv', index=False)

    def split_column_into_rows(self, df, num_columns, overlapping_ratio):
        
        values = df['value'].to_numpy()  

        overlapping_elements = int(num_columns * overlapping_ratio / 100)

        row_elements = num_columns - overlapping_elements

        num_rows = (len(values) + row_elements - 1) // row_elements

        last_row_fill = (num_rows * num_columns) - len(values)

        values = np.pad(values, (0, last_row_fill), mode='constant', constant_values=0)

        new_values = np.zeros((num_rows, num_columns))
        for i in range(num_rows):
            start = i * row_elements
            end = start + num_columns
            new_values[i] = values[start:end]

        new_df = pd.DataFrame(new_values)

        label_name = df.at[0, 'label']
        new_df.insert(0,'label',label_name)

        return new_df
    
    def mergeData(self, sampleSize, overlapRatio):
        self.test_data = pd.DataFrame()
        
        for key in self.datasets.keys():

            df = self.datasets[key]
            df = df.reset_index(drop=True, inplace=False)
            df = self.split_column_into_rows(df,sampleSize, overlapRatio)
            df = df.drop(df.index[-1])
            self.test_data = pd.concat([self.test_data,df])

        self.test_data = self.test_data.reset_index(drop=True, inplace=False)

        print('test data shape: ' , self.test_data.shape)

    def prepareOutput(self):
        
        test_labels = self.test_data['label'].values
        test_data_values = self.test_data.drop(columns={'label'})

        testName = 'test_data_values_' + str(self.sampleSize) + "_" + str(self.overlapRatio)
        labelName = 'test_labels_' + str(self.sampleSize) + "_" + str(self.overlapRatio)

        testName = self.addDatatoPath(testName)
        labelName = self.addDatatoPath(labelName)
        
        test_data_values.to_pickle(testName)
        test_labels = pd.DataFrame(test_labels)
        test_labels.to_pickle(labelName)

        print(len(test_data_values))
        print(test_labels.shape)

        return test_data_values, test_labels
    
    def prepareTestData(self, sampleSize, overlapRatio, path):
        self.sampleSize = sampleSize
        self.overlapRatio = overlapRatio
        self.pickle_path = os.path.join(self.data_save_path, path)
        self.readData(self.pickle_path)
        self.splitData()
        self.mergeData(sampleSize, overlapRatio)
        test_data_values, test_labels = self.prepareOutput()
        return test_data_values, test_labels


import pandas as pd
import numpy as np
import os

class trainData():

    def __init__(self, sampleSize, overlapRatio):

        self.current_folder_path = os.getcwd()
        self.parent_folder_path = os.path.dirname(self.current_folder_path)
        self.data_save_path = os.path.join(self.parent_folder_path, 'data')
        self.sampleSize = sampleSize
        self.overlapRatio = overlapRatio

    def addDataPathtoCurrent(self,path):
        return os.path.join(self.current_folder_path,path)
    
    def readData(self,string):

        df = pd.read_feather(string)

        if "time" in df.columns :
            df = df.drop(columns = {'time'})

        df = df.drop(index=0)


    def splitData(self,df):

        df['label'] = df['label'].str.replace(' ', '')

        # Split the dataset based on labels
        labels = ['Bias', 'Drift', 'Gain', 'NoFault', 'Outliers', 'Precisiondegradation']

        self.datasets = {}
        self.dataframes =[]

        for label in labels:
            
            self.datasets[label] = df[df['label'] == label]

            globals()[f'df_{label}'] = self.datasets[label]
            self.dataframes.append(f'df_{label}')
            #datasets[label].to_csv(self.data_save_path + "/" +f'df_{label}.csv', index=False)

        try:
            print("Size of df_Bias:", df_Bias.shape)
            print("Size of df_Drift:", df_Drift.shape)
            print("Size of df_Gain:", df_Gain.shape)
            print("Size of df_NoFault:", df_NoFault.shape)
            print("Size of df_Outliers:", df_Outliers.shape)
            print("Size of df_PrecisionDegredatation:", df_Precisiondegradation.shape)

        except:
            pass

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
    

    def mergeData(self):
        self.test_data = pd.DataFrame()
        
        for key in self.datasets.keys():

            df = self.datasets[key]
            df = df.reset_index(drop=True, inplace=False)
            df = self.split_column_into_rows(df,self.sampleSize,self.overlapRatio)
            df = df.drop(df.index[-1])
            self.test_data = pd.concat([self.test_data,df])

        self.test_data = self.test_data.reset_index(drop=True, inplace=False)

        print('test data shape: ' + self.test_data.shape)

    def prepareOutput(self):
        test_labels = self.test_data['label'].values
        test_data_values = self.test_data.drop(columns={'label'})
        test_data_values.to_pickle('test_data_values')
        test_labels = pd.DataFrame(test_labels)
        test_labels.to_pickle('test_labels')

        return test_data_values, test_labels
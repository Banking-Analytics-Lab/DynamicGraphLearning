import pandas as pd
class DataBuilder(): 
    def __init__(self,data_path = "", SQL:str = '',con = None) -> None:
        if SQL: 
            self.df = pd.read_sql_query(SQL, con = con)
        elif data_path: 
            self.df = pd.read_csv(data_path,index_col = [0])
            self.df = self.df.iloc[1:1000,0:2]
        
class Node2VectData(DataBuilder): 
    def __init__(self,data_path = "", SQL:str = '',con = None) -> None:
        super().__init__(data_path,SQL,con)
    def build(self,output_file):
        self.df.to_csv(output_file,sep = '\t'  )

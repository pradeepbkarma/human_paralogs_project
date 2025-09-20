

import pandas as pd

class ProcessData:
    def __init__(self, integrated_df, tm_df, ref_list):
        self.orig_data = integrated_df.copy()
        self.tm_data = tm_df.copy()
        self.ref = set(ref_list)
        self.tm_score_merged = self.merge_data()
        self.unique_pairs = self.filter_duplicated_pairs()
        self.filtered_data = self.filter_data()
        self.novel_paralogs = self.get_novel_list()
        
    def merge_data(self):
        merged = self.orig_data.merge(
            self.tm_data, on=['reference', 'target'], how='left'
        )
        return merged 
    
    def filter_duplicated_pairs(self):
        df = self.tm_score_merged.copy()
        df[ 'pairs'] = df.apply(lambda row:tuple(sorted((row['reference'], row['target']))), axis=1)
        data_nodup = df.drop_duplicates("pairs").drop(columns="pairs").reset_index(drop=True)
        return data_nodup
    
    def filter_data(self, threshold=0.5):
        df = self.tm_score_merged.copy()
        mask = (df['ref_tmscore'].fillna(0) >= threshold) | (df['target_tmscore'].fillna(0)>= threshold)
        # filtered_data = df[(df['ref_tmscore']>= 0.5) | (df['target_tmscore'] >=0.5)]
        return df.loc[mask].reset_index(drop=True)
    
    def get_novel_list(self):
        df = self.filtered_data.copy()
        paralogs = set(df['reference'].unique()).union(set(df['target'].unique()))
        novel_list = list(set(paralogs) - self.ref)   
        return novel_list
                
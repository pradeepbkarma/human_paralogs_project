import pandas as pd 

def get_required_df(ref_list, all_data):
    paralogs_data = all_data[all_data['qseqid'].isin(ref_list)]
    # paralogs_data = paralogs_data.drop(columns=['method_p', 'method_mm', 'method_f', 'method_b', 'ensembl', 'prost','mmseqs2', 'foldseek', 'blast'], axis=1)
    paralogs_data.reset_index(drop=True, inplace=True)
    # drop duplicates 
    paralogs_df = paralogs_data.drop_duplicates(subset=['qseqid', 'sseqid'], keep='first', ignore_index=True)
    paralogs_df = paralogs_df.rename(columns={'qseqid':'reference', 'sseqid': 'target'})
    return paralogs_df

# def get_qt_pairs():
    
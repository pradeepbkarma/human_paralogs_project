import pandas as pd 
from collections import Counter
from collections.abc import Iterable

def get_ref_list(path):
    try:
        # try reading with header 
        df = pd.read_csv(path, sep="\t")
        # if first row is header, drop it 
        ref_list = df.iloc[:, 0].dropna().tolist()
        final_list = list(set(ref_list))
    except Exception:
        # if no header, readwithout it 
        df = pd.read_csv(path, sep="\t", header=None)
        ref_list = df.iloc[:, 0].dropna().tolist()
        final_list = list(set(ref_list))
    return final_list

def get_novel_paralogs_df(df, novel_list):
    # remove duplicates by pair 
    data = df.copy()
    data[ 'pairs'] = data.apply(lambda row:tuple(sorted((row['reference'], row['target']))), axis=1)
    data_nodup = data.drop_duplicates("pairs").reset_index(drop=True)

    # filter rows if any item in pairs is in novel_list --> novel paralog pairs 
    novel_set = set(novel_list)
    mask = data_nodup['pairs'].apply(lambda p: any(item in novel_set for item in p))
    
    result = data_nodup[mask].drop(columns='pairs').reset_index(drop=True)
    return result 

def get_novel_list(ref_list, df):
    all_paralogs = list(set(df['reference'].unique()).union(set(df['target'].unique())))
    novel = list(set(all_paralogs) - set(ref_list))
    return novel 

def count_methods(df:pd.DataFrame, 
                  col: str = "methods", 
                  normalize_case: bool = True, 
                  unique_per_row: bool = False) -> dict:
    def to_list(x):
        if pd.isna(x):
            return []
        if isinstance(x, str):
            # allows comma/semicolon 
            parts = x.replace(";", ",").split(",")
            parts = [p.strip() for p in parts if p.strip()]
            return parts
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes, dict)):
            return [str(p).strip() for p in x if str(p).strip()]
        return [str(x).strip()]
    
    methods_list = df[col].apply(to_list)
    if normalize_case:
        methods_list = methods_list.apply(lambda lst: [m.lower() for m in lst])
    
    c = Counter()
    for lst in methods_list:
        c.update(lst)
    return dict(c)




def get_novel_distribution(ref_path, all_data):
    ref_list = get_ref_list(ref_path)
    # filter the data for ref_list

    df = pd.read_csv(all_data, sep="\t")
    filtered_df = df[df['reference'].apply(lambda x: x in ref_list)]
    # filter for the TM-scores 
    tm_filtered = filtered_df[(filtered_df['ref_tmscore'] >= 0.5) & (filtered_df['target_tmscore'] >= 0.5)]
    novel_list = get_novel_list(ref_list, tm_filtered)
    novel_paralogs_df = get_novel_paralogs_df(tm_filtered, novel_list)
    print(f"Total novel paralogs pair is: {len(novel_paralogs_df)}\nTotal novel paralogs is {len(novel_list)}")
    distirbution = count_methods(novel_paralogs_df)
    return distirbution


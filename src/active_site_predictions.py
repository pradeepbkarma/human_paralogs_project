import pandas as pd 
from Bio import SeqIO
import numpy as np 



def get_msa_index(ref_seq: str, ref_seq_pos:list[int])-> list[int]:
    ref_pos_set = set(ref_seq_pos)
    msa_position = []
    seq_index = 0 # 1- based after increment 

    for msa_index, aa in enumerate(ref_seq, start=1):
        if aa != '-':
            seq_index +=1
            if seq_index in ref_pos_set:
                msa_position.append(msa_index)
    
    return msa_position


def get_msa_seq(ref, msa_file_path):
    msa_dict = {}
    records = SeqIO.parse(msa_file_path, 'fasta')
    for record in records:
        msa_dict[record.id] = str(record.seq)
    return msa_dict[ref], msa_dict


def get_active_site_predictions(ref, msa_file_path, ref_pos_list, ref_res_list):
    ref_seq, msa_dict = get_msa_seq(ref, msa_file_path)

    msa_position = get_msa_index(ref_seq, ref_pos_list)
    residues = ref_res_list
    results = []

    for seq_id, seq in msa_dict.items():
        row = {'id': seq_id}
        seq_pos_list = []
        
        # Build mapping: MSA index (1- based) --> sequence index (1-based), skipping gaps
        msa_to_seq_index = {}
        seq_index = 0
        for msa_i, res in enumerate(seq, start=1):
            if res != '-':
                seq_index += 1
                msa_to_seq_index[msa_i] = seq_index
                
        # build inverse mapping for seq index to msa index as well
        seq_index_to_msa = {v:k for k, v in msa_to_seq_index.items()}
        
        counter = 0
        for msa_i in msa_position:
            active_res = residues[counter]
            res = seq[msa_i -1 ]
            # print(res)
            
            if res == active_res:
                row[f'msa_{msa_i}'] = res
                seq_pos_list.append(msa_to_seq_index.get(msa_i, None))
            else:
                # search upstream and downstream +- 20
                found_match = False
                current_seq_pos = msa_to_seq_index.get(msa_i, None)
                if current_seq_pos:
                    for offset in range(1, 21):
                        for direction in [-1, 1]:
                            new_seq_pos = current_seq_pos + direction*offset
                            if new_seq_pos in seq_index_to_msa:
                                candidate_msa_i = seq_index_to_msa[new_seq_pos]
                                if seq[candidate_msa_i -1] == active_res:
                                    row[f'msa_{msa_i}'] = f"{active_res}_{candidate_msa_i}"
                                    seq_pos_list.append(new_seq_pos)
                                    found_match = True
                                    break
                        if found_match:
                            break
                if not found_match:
                    # look for two resides upstream and downstream from the current position 
                    putative_match_res = []
                    putative_match_pos = []
                    if current_seq_pos is None:
                        # Gap at the MSA column 
                        up_found = 0
                        j = 1
                        while up_found <2 and (msa_i - j) >=1:
                            if seq[msa_i -j -1] != '-':
                                seq_pos = msa_to_seq_index[msa_i -j]
                                putative_match_res.append(seq[msa_i-j-1])
                                putative_match_pos.append(seq_pos)
                                up_found += 1
                            j += 1
                        # Downstream MSA 
                        down_found = 0
                        j = 1
                        while down_found <2 and (msa_i +j) <= len(seq):
                            if seq[msa_i + j -1] != '-':
                                seq_pos = msa_to_seq_index[msa_i + j]
                                putative_match_res.append(seq[msa_i + j -1])
                                putative_match_pos.append(seq_pos)
                                down_found += 1
                            j += 1

                    else:
                        putative_match_res.append(res)
                        putative_match_pos.append(current_seq_pos)
                        for direction in (-1,1):
                            # for offset in (1,2):
                            new_seq_pos = current_seq_pos+direction
                            msa_j = seq_index_to_msa.get(new_seq_pos)
                            if msa_j:
                                putative_match_res.append(seq[msa_j -1])
                                putative_match_pos.append(new_seq_pos)

                    row[f'msa_{msa_i}'] = putative_match_res
                    seq_pos_list.append(putative_match_pos)
                    # seq_pos_list.append(current_seq_pos)
            
            counter += 1
            
        row['seq_pos'] = seq_pos_list
        results.append(row)
    df_out = pd.DataFrame(results)
    return df_out












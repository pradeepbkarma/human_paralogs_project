import pandas as pd 


def get_tm_aln_df(aln_data):
    # convert the unstructured tm-align results into structured dataframe 
    reference= []
    target = []
    ref_res = []
    target_res = []
    length = []
    tm_rmsd = []
    ref_tmscore = []
    target_tmscore = []

    with open(aln_data, 'r') as f:
        ref = t = None
        ref_len = target_len = aln_len = rmsd = ref_tm = target_tm = None
        for line in f:
            line = line.strip()
            if line.startswith("Name of Chain_1"):
                ref=line.split("/")[-1].split(".")[0]
                # print(ref) # debug logging to see if it's printing the expected result

            elif line.startswith("Name of Chain_2"):
                t = line.split("/")[-1].split(".")[0]
                # print(t)
            
            elif line.startswith("Length of Chain_1"):
                ref_len = int(line.split(":")[1].split()[0])
            elif line.startswith("Length of Chain_2"):
                target_len = int(line.split(":")[1].split()[0])
            elif line.startswith("Aligned length="):
                parts = line.split(",")
                aln_len = int(parts[0].split("=")[1].strip())
                rmsd = float(parts[1].split("=")[1].strip())
            
            elif line.startswith("TM-score=") and "normalized by length of Chain_1" in line:
                ref_tm = float(line.split("=")[1].split()[0])
            elif line.startswith("TM-score=") and "normalized by length of Chain_2" in line:
                target_tm = float(line.split("=")[1].split()[0])
            elif line.startswith("Total CPU time"):
                # Only append if all variables are set
                if None not in (ref, t, ref_len, target_len, aln_len, rmsd, ref_tm, target_tm):
                    reference.append(ref)
                    target.append(t)
                    ref_res.append(ref_len)
                    target_res.append(target_len)
                    length.append(aln_len)
                    tm_rmsd.append(rmsd)
                    ref_tmscore.append(ref_tm)
                    target_tmscore.append(target_tm)
    tm_data = pd.DataFrame({
        "reference": reference,
        "target": target,
        "ref_len": ref_res,
        "target_len":target_res,
        "tm_rmsd": tm_rmsd,
        "ref_tmscore": ref_tmscore,
        "target_tmscore": target_tmscore,
        "aln_len": length
    })
    return tm_data

This integrated homolog indentifying framework uses sequence-based tools BLAST, and MMseqs2; 
structure-based tool Foldseek, and protein language model based tool PROST. 
All the homologs after all-vs-all exhaustive search for each tool and reciprocal hits filteration is 
given in integrated_paralogs_methods.tsv. 

Follow the steps used in integrated_paralogs_framework.ipynb to get the putative human paralogs for provided 
list of proteins of interests as reference set. 

Follow the steps used in functional_predictions.ipynb to predict active site residues for 
structurally similar proteins based on the active site residue informations of previously 
annotated reference protein. 

reproduce the virtual environment as 
conda create -f human_paralogs_environment.yml

Follow the FoldMason github page to install the tool @ https://github.com/steineggerlab/foldmason 

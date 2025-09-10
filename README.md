# PLM-pipeline

The pipeline takes as input a list of sequences in a CSV, which are passed into a pretrained language model to perform downstream calculations.

## Set up environment

- Create a new environment:
`conda create -n plm`

- Activate the environment:
`conda activate plm`

- To install necessary libraries:
`pip install -r requirements.txt`  
or `pip install -r requirements_sapiens.txt` since Sapiens is not compatible with Python version 3.9 and above.

## Run the pipeline
(optional) Set the cache directory in the shell: `export HF_HOME="/path/to/your/cache_dir"`  
Run the script `scripts/pipeline.py ` with the following arguments:
- `model_name` class name of the model to use. Currently available: Ablang, Ablang2, ProtBert, Sapiens, ESM1b, ESMC, ESM2
- `file_path` path with the location of the input CSV file.
    - This CSV file should contain:
        - a column with a unique sequence identifier specified in the parameter `sequence_id_column`
        - a column with the sequence specified in the parameter `sequence_column`
        - a column named `chain` with either IGH/IGL/IGK for Ig heavy chain or light chain sequences respectively, only necessary when using the models Ablang and Sapiens   
    - When using Ablang2, the VH and VL sequences of the same cells should have matching `sequence_id_column`
- `sequence_column` column name of the sequence to use in the input CSV
- `sequence_id_column` column name of the sequence identifier in the input CSV (default: sequence_id)
- `output_folder` folder name to store output
- `calc_list` list of the downstream calculations to perform
    - `pseudolikelihood` calculates pseudolikelihood for each sequence, adds it to the input CSV, and saves this as output CSV
    - `probability_matrix` calculates the probability matrix of each amino acid per position of each sequence and save this matrix as CSV
    - `embeddings` calculate the embeddings per sequence or per residue based on the parameter `embeddings_method`
    - `suggest_mutations`: based on the previously calculated probability matrix values it suggests a desired number of mutations
    - `attention_matrix` calculates the attention matrix between residues in a sequence
- `number_mutations` number of mutations you want the model to suggest (default is 1)
- `cache_dir` path to a cache directory to download the pretrained ESM models if the standard cache should not be used.
- `embeddings_method` method to output the embeddings, one of:
    - `average_pooling` takes the average output embeddings for each sequence, adds it to the input CSV, and saves this as output CSV (default)
    - `per_token` takes the embedding per residue, outputs a CSV per sequence
- `attention_head` how to pool the attention heads per layer, one of:
    - `average` (default)
    - `sum`
- `layer` for which layer(s) to get the embedding or attention (default is last)
    - work in progress...


## Adding new models
- Create a .py file which will contain the model class in the folder `src/`. Be careful that the name of the file is not the same as any of the packages that we are using
- Make a model class file like the others (simple example is the ablang_model). 
- Each class consists of a init function, where you initialize things, like first making the wanted model and adding it to the self.model variable. 
- Then it contains functions for downstream calculations:
    - `fit_transform` which will transform the sequences using the model and create embeddings.
    - `calc_pseudo_likelihood_sequence` calculates the pseudolikelihood of whole sequence (sum of log-scaled per-residue probablities, divided by sequence length)
    - `calc_probability_matrix` calculates the probability matrix of each amino acid per position of a sequence
    - `calc_attention_matrix` calculates the attention matrix between residues in a sequence
- Add the model to `scripts/pipeline.py` as import and at the initialization step.

## Adding new functions
- Add the function to each model class file in the folder `src/`
- General functions can be written in `src/utils.py`
- Call the new function in the `scripts/pipeline.py` script seperately for Ablang/Sapiens and other models
- Add the new function plus a description in this README

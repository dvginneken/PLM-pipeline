import ablang2
import pandas as pd
import scipy
import sys
import torch
import os

sys.path.append("../scripts")

class Ablang2():

    """
    Class for the protein Model Ablang2
    """

    def __init__(self):
        """
        Initializes the Ablang2 model.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Download and initialise the model
        self.model = ablang2.pretrained(model_to_use='ablang2-paired', device=self.device)

        #dont update the weights
        self.model.freeze()

    


    def fit_transform(self, sequence_file, layer:str = "last", method:str = "average_pooling", save_path:str = ".", model_name:str = "ESMc", 
                      seq_id_column:str = "sequence_id", sequences_column:str = "sequence"):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequence_file: `dataframe` 
        DataFrame with sequences to be transformed
        
        layer: `str`
        Layer from which to extract the embeddings. Default is "last".

        method: `str`
        Method to extract the embeddings. Default is "average_pooling". Options: "average_pooling", "per_token".

        save_path: `str`
        Path to save the embeddings CSV file. Default is current directory.

        model_name: `str`
        Name of the model, used for saving the embeddings file. Default is "ESMc".

        seq_id_column: `str`
        Column name in the sequence_file DataFrame that contains unique sequence identifiers. Default is "sequence_id".

        sequences_column: `str`
        Column name in the sequence_file DataFrame that contains sequences. Default is "sequence".


        """
        all_seqs = [sequence.split("|") for sequence in sequence_file[sequences_column]]
        if method == "average_pooling": #The embeddings are made my averaging across all residues
            output = self.model(all_seqs, mode="seqcoding")
            embeds = pd.DataFrame(output,columns=[f"dim_{i}" for i in range(output.shape[1])])
            embeds.to_csv(os.path.join(save_path,f"embeddings_{model_name}.csv"), index=False)
        elif method == "per_token": #The embeddings are made by extracting the last layer of the model
            for index, row in sequence_file.iterrows():
                sequence = row[sequences_column].split("|")
                seq_id = row[seq_id_column]
                output = self.model(sequence, mode="rescoding")[0]
                embeds = pd.DataFrame(output, columns=[f"dim_{i}" for i in range(output.shape[1])])
                embeds.to_csv(os.path.join(save_path,f"embeddings_seq_{seq_id}_{model_name}.csv"), index = False)
        
        

    def calc_pseudo_likelihood_sequence(self, sequences: list):
        """
        Calculate the pseudolikelihood of a list of sequences.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed, VH and VL are separated with |
        ------

        pll_all_sequences: `array`
        """
        all_seqs = [sequence.split("|") for sequence in sequences]  # Split sequences in VH an VL 
        pll_all_sequences = self.model(all_seqs, mode='pseudo_log_likelihood') # Calculate pseudolikelihood
        return pll_all_sequences

    def calc_probability_matrix(self, sequence:str):
        """
        Calculate the probability matrix of the heavy and light chain sequences.
        
        parameters
        ----------

        sequence: `string` 
        Sequences to be transformed, VH and VL are separated with |
        ------

        prob_matrix: `dataframe`
        Probability matrix of each amino acid in this VH+VL sequence
        """
        sequence = [sequence.split("|")] # Split sequence in VH an VL
        logits = self.model(sequence, mode="likelihood")[0] # Calculate the likelihood
        prob = scipy.special.softmax(logits,axis = 1) # Softmax transformation
        prob_matrix = pd.DataFrame(prob, columns = list(self.model.tokenizer.decode(range(0,26)))).iloc[1:-4,] # Transform to dataframe and remove the start, stop and sep positions
        prob_matrix = prob_matrix.drop(columns=['<','>','*','X','|','-']) # Drop special tokens
        prob_matrix = prob_matrix.reindex(sorted(prob_matrix.columns), axis=1) # Sort columns on alphabetical order
        return prob_matrix
    
    def calc_attention_matrix(self, sequence:str, layer:str = "last", head:str = "average"):
        """
        Calculates the attention matrix for a given sequence and layer.

        parameters
        ----------
        sequence: `str`
        The input protein sequence.

        layer: `int`
        The layer from which to extract the attention scores. Default is -1 (last layer).

        head: `str`
        The attention head to extract scores from. Default is "average".

        returns
        -------
        attn_matrix: `DataFrame`
        A DataFrame containing the attention matrix for the sequence.s
        """
        print("Attention matrices for Ablang2 are not implemented yet.")
        
        df = pd.DataFrame()

        return df
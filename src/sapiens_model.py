import sapiens
import sys
import pandas as pd
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm

sys.path.append(os.getcwd()+"/src")
class Sapiens():

    """
    Class for the protein Model Sapiens
    Author: Aurora
    """

    def __init__(self, chain_type="H"):
        """
        Creates the instance of the language model instance

        parameters
        ----------

        chain_type: `str`
        `L` or `H` whether the input is from light or heavy chains resprectively
        
        method: `str`
        Layer that we want the embedings from

        file_name: `str`
        The name of the folder to store the embeddings
        """

        self.chain = chain_type

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

        ----------
        return: `dataframe`
        DataFrame with the extracted embeddings if method is "average_pooling".

        """
        pooler_zero = np.zeros((len(sequence_file.index),128))
        for index, row in sequence_file.iterrows():
            sequence = row[sequences_column]
            seq_id = row[seq_id_column]
   
            if method == "average_pooling":
                if layer == "last":
                    embeddings_output = sapiens.predict_sequence_embedding(sequence, chain_type=self.chain, layer = None)[-1] # Get the embeddings of the last hidden layer
                #TO DO
                # if layer == ..
                # 
                pooler_zero[index,:] = embeddings_output.tolist()

            elif method == "per_token": # Per token embeddings
                if layer == "last":
                    embeddings_output = sapiens.predict_residue_embedding(sequence, chain_type=self.chain, layer = None)[-1] # Get the embeddings of the last hidden layer
                #TO DO
                # if layer == ..
                # 
                embeds = pd.DataFrame(embeddings_output, columns=[f"dim_{i}" for i in range(embeddings_output.shape[1])])
                embeds.to_csv(os.path.join(save_path,f"embeddings_seq_{seq_id}_{model_name}.csv"), index = False)

                # Save the average embeddings to a CSV file
        if method == "average_pooling":
            embeds = pd.DataFrame(pooler_zero,columns=[f"dim_{i}" for i in range(pooler_zero.shape[1])])
            embeds = pd.concat([sequence_file,embeds],axis=1) # Add to the sequence file 
            return embeds

    def calc_pseudo_likelihood_sequence(self, sequences:list):
        pll_all_sequences = []
        for j,sequence in enumerate(tqdm(sequences)):
            try:
                amino_acids = list(sequence)
                df = pd.DataFrame(sapiens.predict_scores(sequence, chain_type=self.chain))

                per_position_ll = []
                for i in range(len(amino_acids)):
                    aa_i = amino_acids[i]
                    if aa_i == "-" or aa_i == "*":
                        continue
                    ll_i = np.log(df.iloc[i,:][aa_i])
                    per_position_ll.append(ll_i)
                
                pll_seq = np.average(per_position_ll)
                pll_all_sequences.append(pll_seq)
            except:
                pll_all_sequences.append(None)

        return pll_all_sequences
    
    def calc_probability_matrix(self, sequence:str):
        df = pd.DataFrame(sapiens.predict_scores(sequence, chain_type=self.chain))

        return df
import ablang
import numpy as np
import pandas as pd
import pickle as pkl
import scipy
from tqdm import tqdm
import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append("../scripts")

class Ablang():

    """
    Class for the protein Model Ablang
    """

    def __init__(self, chain = "heavy", calc_list = None, cache_dir = "default"):
        """
        Creates the instance of the language model instance; either light or heavy

        parameters
        ----------
        chain: `str`
        The chain type to use for the model. Options are "heavy" or "light".

        calc_list: `list`
        List of calculations to perform with the model.

        cache_dir: `str`
        The directory to use for caching model files. Default is "default".
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ablang.pretrained(chain,device=self.device)

        if "attention_matrix" in calc_list:
            if chain == "heavy":
                model_name = 'qilowoq/AbLang_heavy'
            elif chain == "light":
                model_name = 'qilowoq/AbLang_light'

            if cache_dir != "default":
                CACHE_DIR = cache_dir
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
                self.attention_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_attentions=True, attn_implementation="eager", cache_dir=CACHE_DIR).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.attention_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_attentions=True, attn_implementation="eager").to(self.device)

    


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
        sequences = sequence_file[sequences_column].tolist()
        if method == "average_pooling": #The embeddings are made my averaging across all residues 
            output = self.model(sequences, mode="seqcoding")  
            return pd.DataFrame(output,columns=[f"dim_{i}" for i in range(output.shape[1])])
        elif method == "per_token": #The embeddings are made by extracting the last layer of the model
            for index, row in sequence_file.iterrows():
                sequence = row[sequences_column]
                seq_id = row[seq_id_column]
                output = self.model(sequence, mode="rescoding")[0]
                embeds = pd.DataFrame(output, columns=[f"dim_{i}" for i in range(output.shape[1])])
                embeds.to_csv(os.path.join(save_path,f"embeddings_seq_{seq_id}_{model_name}.csv"), index = False)

                

    def calc_pseudo_likelihood_sequence(self, sequences: list):
        """
        Calculates the pseudo-likelihood for a list of sequences.

        parameters
        ----------
        sequences: `list`
        List of sequences to calculate the pseudo-likelihood for.

        ----------
        return: `list`
        List of pseudo-likelihood values for each sequence.
        """
        pll_all_sequences = []
        for j,sequence in enumerate(tqdm(sequences)):
            try:
                amino_acids = list(sequence)
                logits = self.model(sequence, mode="likelihood")[0]
                prob = scipy.special.softmax(logits,axis = 1)
                df = pd.DataFrame(prob, columns = list(self.model.tokenizer.vocab_to_aa.values())[4:])
                df = df.iloc[1:-1,:]
                df = df.reindex(sorted(df.columns), axis=1)


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
        """
        Calculates the probability matrix for a given sequence.

        parameters
        ----------
        sequence: `str`
        The input sequence to calculate the probability matrix for.

        ----------
        return: `dataframe`
        DataFrame containing the probability matrix for the input sequence.
        """
        logits = self.model(sequence, mode="likelihood")[0]
        prob = scipy.special.softmax(logits,axis = 1)
        df = pd.DataFrame(prob, columns = list(self.model.tokenizer.vocab_to_aa.values())[4:])
        df = df.iloc[1:-1,:]
        df = df.reindex(sorted(df.columns), axis=1)

        return df
    
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
        The attention head to extract scores from ("average" or "sum"). Default is "average".

        returns
        -------
        attn_matrix: `DataFrame`
        A DataFrame containing the attention matrix for the sequence.
        """
        amino_acids = list(sequence)
        seq_tokens = ' '.join(amino_acids)
        seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
        seq_tokens = seq_tokens.to(self.device)
        outputs = self.attention_model(**seq_tokens, output_attentions=True)
        if layer == "last":
            layer_int = -1
        attn_scores = outputs.attentions[1][layer_int] #batch 1 and selected layer
        if head == "average":
            attn_matrix = attn_scores.mean(dim=0)
        elif head == "sum":
            attn_matrix = attn_scores.sum(dim=0)
        df = pd.DataFrame(attn_matrix.cpu().detach().numpy()).iloc[1:-1, 1:-1]

        return df
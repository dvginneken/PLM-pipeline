from typing import Union

import sapiens
import pandas as pd
import numpy as np
from tqdm import tqdm


class Sapiens:
    """
    Class for the protein Model Sapiens
    Author: Aurora
    """

    def __init__(self, chain_type: str = "H", method: str = "average", file_name: str = "."):
        """
        Creates the instance of the language model instance

        parameters
        ----------

        chain_type: `str`
        `L` or `H` whether the input is from light or heavy chains respectively

        method: `str`
        Layer that we want the embeddings from

        file_name: `str`
        The name of the folder to store the embeddings
        """

        self.chain = chain_type
        self.file = file_name

    def fit_transform(self, sequences: list) -> Union[pd.DataFrame, None]:
        """
        Fits the model and outputs the embeddings.

        parameters
        ----------

        sequences: `list`
        Column with sequences to be transformed
        ------

        None, saved the embeddings in the embeddings.csv
        """
        print("Using the average layer")
        output = []
        for sequence in sequences:
            try:
                result = np.mean(
                    np.mean(sapiens.predict_residue_embedding(sequence, chain_type=self.chain), axis=1),
                    axis=0,
                )
                output.append(dict(enumerate(result)))
            except Exception as e:
                print(e)
                continue
        output = pd.DataFrame(output).add_prefix("dim_")
        return output.reset_index(drop=True)

    def calc_pseudo_likelihood_sequence(self, sequences: list) -> list:
        pll_all_sequences = []
        for sequence in tqdm(sequences):
            try:
                amino_acids = list(sequence)
                df = pd.DataFrame(sapiens.predict_scores(sequence, chain_type=self.chain))
                per_position_ll = []
                for i, amino_acid in enumerate(amino_acids):
                    ll_i = np.log(df.iloc[i, :][amino_acid])
                    per_position_ll.append(ll_i)

                pll_seq = np.average(per_position_ll)
                pll_all_sequences.append(pll_seq)
            except Exception as e:
                print(e)
                pll_all_sequences.append(None)
        return pll_all_sequences

    def calc_probability_matrix(self, sequence: str) -> pd.DataFrame:
        df = pd.DataFrame(sapiens.predict_scores(sequence, chain_type=self.chain))
        return df

import sapiens
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm

from .utils import get_pseudo_likelihood


class Sapiens:
    """
    Class for the protein Model Sapiens
    Author: Aurora
    """

    def __init__(self, chain_type="H", method="average", file_name="."):
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
        if isinstance(method, int):
            self.layer = method
        elif method == "average":
            self.layer = None
        else:
            self.layer = "prob"
        self.file = file_name

    def fit_transform(self, sequences):
        """
        Fits the model and outputs the embeddings.

        parameters
        ----------

        sequences: `list`
        Column with sequences to be transformed
        ------

        None, saved the embeddings in the embeddings.csv
        """

        if self.layer == None:
            print("Using the average layer")
            output = []
            for j, sequence in enumerate(sequences):
                try:
                    output.append(
                        list(
                            np.mean(
                                np.mean(sapiens.predict_residue_embedding(sequence, chain_type=self.chain), axis=1),
                                axis=0,
                            )
                        )
                    )
                except Exception as e:
                    print(e)
                    continue
            output = pd.DataFrame(output, columns=[f"dim_{i}" for i in range(len(output[0]))])
            return output.reset_index(drop=True)
        elif self.layer == "prob":
            print("\n Making probabilities")
            output = sequences.apply(lambda seq: pd.DataFrame(sapiens.predict_scores(seq, chain_type=self.chain)))
            embedings = get_pseudo_likelihood(output, sequences)
            pkl.dump([output, embedings], open("outfiles/" + self.file + "/probabilities_pseudo.pkl", "wb"))
        else:
            print("\nUsing the {} layer".format(self.layer))
            output = []
            for j, sequence in enumerate(sequences):
                try:
                    output.append(
                        list(
                            np.mean(sapiens.predict_residue_embedding(sequence, chain_type=self.chain), axis=1)[
                                self.layer - 1, :
                            ]
                        )
                    )
                except:
                    continue
            output.columns = [f"dim_{i}" for i in range(output.shape[1])]
            return output.reset_index(drop=True)

    def calc_pseudo_likelihood_sequence(self, sequences: list):
        pll_all_sequences = []
        for j, sequence in enumerate(tqdm(sequences)):
            try:
                amino_acids = list(sequence)
                df = pd.DataFrame(sapiens.predict_scores(sequence, chain_type=self.chain))

                per_position_ll = []
                for i in range(len(amino_acids)):
                    aa_i = amino_acids[i]
                    if aa_i == "-" or aa_i == "*":
                        continue
                    ll_i = np.log(df.iloc[i, :][aa_i])
                    per_position_ll.append(ll_i)

                pll_seq = np.average(per_position_ll)
                pll_all_sequences.append(pll_seq)
            except:
                pll_all_sequences.append(None)

        return pll_all_sequences

    def calc_probability_matrix(self, sequence: str):
        df = pd.DataFrame(sapiens.predict_scores(sequence, chain_type=self.chain))

        return df

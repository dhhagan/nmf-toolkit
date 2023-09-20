import unittest
import nmf_toolkit
from nmf_toolkit.models import NMF
from nmf_toolkit.figures import cv_figure, rankcomp_figure
import numpy as np
import pandas as pd


def make_dataset(N=150, R=4):
    noise = 0.8
    replicates = 10
    ranks = np.arange(1, 6)

    # Init
    U = np.random.rand(N, R)
    Vt = np.random.rand(R, N)

    # Make the data noisy
    data = np.dot(U, Vt) + noise * np.random.rand(N, N)
    
    df = pd.DataFrame(data, columns=[f"C{i}" for i in range(N)])

    return df


class TestClass(unittest.TestCase):
    
    # def test_cv(self):
    #     """Test computing the cross validation
    #     """
    #     # Make a fake dataset
    #     data = make_dataset(N=150, R=4)
        
    #     # Initialize an NMF object
    #     obj = NMF(data=data)
        
    #     # Find the optimal rank
    #     rnk, rv = obj.find_rank(max_rank=6, replicates=10)
        
    #     self.assertEqual(rnk, 4)
        
    #     # Make the figure
    #     fig = cv_figure(rv)
        
    #     fig.savefig("tmp.png", dpi=350)
        
    def test_fit_model(self):
        """_summary_
        """
        data = make_dataset(N=150, R=4)
        
        # Init
        obj = NMF(data=data)
        
        # Fit
        rv, comp, res = obj.fit()
        
        
    def test_bootstrap(self):
        """_summary_
        """
        data = make_dataset(N=10, R=4)
        
        # Init
        obj = NMF(data=data)
        
        # Run a bootstrap analysis
        rv = obj.bootstrap(n_iter=25, rank=4, frac=0.25)
        
        # Make a figure
        fig = rankcomp_figure(rv)
        
        fig.savefig("tmp-rankfig.png", dpi=350)
import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import NMF as _NMF
from nmf_toolkit.utils import cv_nmf


class NMF(object):
    def __init__(self, **kwargs):
        self.data = kwargs.pop("data", None)
        
    def _fit(self, data, rank=3, **kwargs):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        rank : int, optional
            _description_, by default 3
        """
        # Here, we could validate the data
        
        # Set up the model
        nmf = _NMF(n_components=rank, alpha_W=0.1, max_iter=250)
        
        # Fit the data
        W = nmf.fit_transform(X=data.T)
        H = nmf.components_
        
        # Convert the results to a DataFrame
        # idx = self.data.index if self.data.index.dtype == "datetime64[ns]" else None
        if type(self.data) is pd.DataFrame:
            idx = data.index if data.index.dtype == "datetime64[ns]" else None
        else:
            idx = None
            
        tseries = pd.DataFrame(
            H.T, 
            index=idx,
            columns=[f"F{i}" for i in range(H.T.shape[1])]
        )
        
        # Calculate the composition
        comp = pd.DataFrame(W.T, index=tseries.columns, columns=data.columns)
        
        print (H)
        # Compute the residual for each column
        res = list()
        print (comp)
        for c in comp.columns:
            # Compute the total sum of the column
            bf = pd.DataFrame(comp[c].values * H.T).sum()
            
            # Normalize to the total amount
            bf = bf / data[c].sum()
            
            res.append(pd.DataFrame(bf, columns=[c]).T)
        
        # Concat
        res = pd.concat(res)
        res.columns = tseries.columns
        res['Residual'] = 1 - res.sum(axis=1)
            
        return tseries, comp, res
        
    def fit(self, rank=3, **kwargs):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        rank : int, optional
            _description_, by default 3
        """
        return self._fit(data=self.data, rank=rank, **kwargs)
        
    def find_rank(self, max_rank=6, replicates=3, **kwargs):
        """_summary_
        """
        rv = list()
        
        for rnk, j in itertools.product(np.arange(1, max_rank), range(replicates)):
            train, test, conv = cv_nmf(
                self.data.values,
                rank=rnk,
                verbose=False,
                tol=1e-4,
                max_iter=150,
                p_holdout=0.2
            )[2:]
            
            rv.append(dict(Rank=int(rnk), MSE=test, Group="Test", Converged=conv))
            rv.append(dict(Rank=int(rnk), MSE=train, Group="Train", Converged=conv))
        
        rv = pd.DataFrame(rv)
        
        # Group 
        rv = rv.groupby(["Group", "Rank"]).describe(percentiles=[0.05, 0.95])
        
        # Find the ideal rank
        ideal_rank = (rv["MSE"]["mean"]["Test"] > rv["MSE"]["mean"]['Test'].shift()).idxmax() - 1
            
        return ideal_rank, rv
    
    def bootstrap(self, rank=3, n_iter=5, frac=0.1, **kwargs):
        """_summary_

        Parameters
        ----------
        n_iter : int, optional
            The number of runs, by default 5
        frac : float, optional
            The percentage of data to use in each run, by default 0.1
        """
        # Create holders for the results
        rv = list()
        
        # Iterate and generate the data
        for i in range(n_iter):
            # Randomly select a subset of the data
            df = self.data.sample(frac=frac)
            
            # Fit the data
            _, comp, res = self._fit(data=df, rank=rank, **kwargs)
            
            rv.append(res)
            
        rv = pd.concat(rv, sort=False)
            
        return rv.reset_index().melt(id_vars=['index'])
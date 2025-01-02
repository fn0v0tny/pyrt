#!/usr/bin/python3
"""
termfit.py
generic fitter for models with terms
(c) Martin Jelinek, ASU AV CR, 2021-2023
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize as fit
from astropy.table import Table
from numpy.typing import NDArray


class termfit:
    """Fit data with string identified terms"""

    def __init__(self) -> None:
        self.delin: bool = False
        self.fitterms: List[str] = []
        self.fitvalues: List[float] = []
        self.fiterrors: List[float] = []
        self.fixterms: List[str] = []
        self.fixvalues: List[float] = []
        self.sigma: float = np.nan
        self.variance: float = np.nan
        self.ndf: float = np.nan
        self.wssr: float = np.nan
        self.wssrndf: float = np.nan
        self.modelname: str = "Model"

    def fixall(self) -> None:
        """Fix all terms in the model."""
        self.fixterms = self.fixterms + self.fitterms
        self.fixvalues = self.fixvalues + self.fitvalues
        self.fitterms = []
        self.fitvalues = []

    def fixterm(self, terms: List[str], values: Optional[List[float]] = None) -> None:
        """Add and fixate a term.
        
        Args:
            terms: List of term names to fix
            values: Optional list of values for the terms
        """
        if values is None:
            xvalues: List[float] = []
            for term in terms:
                for ft, fv in zip(
                    self.fixterms + self.fitterms, self.fixvalues + self.fitvalues
                ):
                    if ft == term:
                        xvalues += [fv]
        else:
            xvalues = values

        for term, value in zip(terms, xvalues):
            newft: List[str] = []
            newfv: List[float] = []
            for ft, fv in zip(self.fitterms, self.fitvalues):
                if ft != term:
                    newft += [ft]
                    newfv += [fv]
            self.fitterms = newft
            self.fitvalues = newfv
            newft = []
            newfv = []
            for ft, fv in zip(self.fixterms, self.fixvalues):
                if ft != term:
                    newft += [ft]
                    newfv += [fv]
            self.fixterms = newft + [term]
            self.fixvalues = newfv + [value]

    def fitterm(self, terms: List[str], values: Optional[List[float]] = None) -> None:
        """Add and set a term to be fitted.
        
        Args:
            terms: List of term names to fit
            values: Optional list of initial values for the terms
        """
        if values is None:
            xvalues: List[float] = []
            for term in terms:
                for ft, fv in zip(
                    self.fixterms + self.fitterms, self.fixvalues + self.fitvalues
                ):
                    if ft == term:
                        xvalues += [fv]
        else:
            xvalues = values

        for term, value in zip(terms, xvalues):
            newft: List[str] = []
            newfv: List[float] = []
            for ft, fv in zip(self.fitterms, self.fitvalues):
                if ft != term:
                    newft += [ft]
                    newfv += [fv]
            self.fitterms = newft + [term]
            self.fitvalues = newfv + [value]
            newft = []
            newfv = []
            for ft, fv in zip(self.fixterms, self.fixvalues):
                if ft != term:
                    newft += [ft]
                    newfv += [fv]
            self.fixterms = newft
            self.fixvalues = newfv

    def termval(self, term: str) -> float:
        """Return value of a term in question.
        
        Args:
            term: The term name to look up
            
        Returns:
            The value of the term, or np.nan if not found
        """
        for ft, fv in zip(
            self.fixterms + self.fitterms, self.fixvalues + self.fitvalues
        ):
            if ft == term:
                return fv
        return np.nan

    def __str__(self) -> str:
        """Print all terms fitted by this class."""
        output: str = ""
        for term, value in zip(self.fixterms, self.fixvalues):
            output += "%-8s= %16f / fixed\n" % (term, value)

        i = 0
        for term, value in zip(self.fitterms, self.fitvalues):
            try:
                error = self.fiterrors[i]
            except IndexError:
                error = np.nan
            output += "%-8s= %16f / ± %f (%.3f%%)\n" % (
                term,
                value,
                error,
                np.abs(100 * error / value),
            )
            i += 1
        output += "NDF     = %d\n" % (self.ndf)
        output += "SIGMA   = %.3f\n" % (self.sigma)
        output += "VARIANCE= %.3f\n" % (self.variance)
        output += "WSSR/NDF= %.3f" % (self.wssrndf)

        return output

    def oneline(self) -> str:
        """Print all terms fitted by this class in a single line that can be loaded later."""
        output: str = ""
        comma: bool = False
        for term, value in zip(
            self.fixterms + self.fitterms, self.fixvalues + self.fitvalues
        ):
            if comma:
                output += ","
            else:
                comma = True
            output += "%s=%f" % (term, value)
        return output

    def fit(self, data: List[NDArray[np.float64]]) -> fit.OptimizeResult:
        """Fit data to the defined model.
        
        Args:
            data: List of numpy arrays containing the data to fit
            
        Returns:
            The optimization result from scipy.optimize.least_squares
        """
        self.delin = False
        res = fit.least_squares(self.residuals, self.fitvalues, args=[data], ftol=1e-14)
        self.fitvalues = []
        for x in res.x:
            self.fitvalues += [x]
        self.ndf = len(data[0]) - len(self.fitvalues)
        self.wssr = np.sum(self.residuals(self.fitvalues, data))
        self.sigma = np.median(self.residuals0(self.fitvalues, data)) / 0.67
        self.variance = np.median(self.residuals(self.fitvalues, data)) / 0.67
        self.wssrndf = self.wssr / self.ndf

        try:
            cov = np.linalg.inv(res.jac.T.dot(res.jac))
            self.fiterrors = np.sqrt(np.diagonal(cov))
        except:
            self.fiterrors = res.x * np.nan
        return res

    def cauchy_delin(
        self, arg: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Cauchy delinearization to give outliers less weight and have more robust fitting.
        
        Args:
            arg: Input value(s) to delinearize
            
        Returns:
            Delinearized value(s)
        """
        try:
            ret = np.sqrt(np.log1p(arg ** 2))
        except RuntimeWarning:
            print(str(arg))
            ret = np.sqrt(np.log1p(arg ** 2))
        return ret

    def savemodel(self, file: str) -> None:
        """Write model parameters into an ecsv file.
        
        Args:
            file: Path to the output file
        """
        errs: List[float] = []
        i = 0
        for term in self.fitterms + self.fixterms:
            try:
                e = self.fiterrors[i]
            except IndexError:  # fixed, not yet fitted etc.
                e = 0
            errs += [e]
            i += 1

        amodel = Table(
            [
                self.fitterms + self.fixterms,
                self.fitvalues[0 : len(self.fitterms)] + self.fixvalues,
                errs,
            ],
            names=["term", "val", "err"],
        )
        amodel.meta["name"] = self.modelname
        amodel.meta["sigma"] = self.sigma
        amodel.meta["variance"] = self.variance
        amodel.meta["wssrndf"] = self.wssrndf
        amodel.write(file, format="ascii.ecsv", overwrite=True)

    def readmodel(self, file: str) -> None:
        """Read model parameters from an ecsv file.
        
        Args:
            file: Path to the input file
        """
        self.fixterms = []
        self.fixvalues = []
        self.fitterms = []
        self.fitvalues = []
        self.fiterrors = []

        rmodel = Table.read(file, format="ascii.ecsv")

        for param in rmodel:
            if param["err"] == 0:
                self.fixterms += [param["term"]]
                self.fixvalues += [param["val"]]
            else:
                self.fitterms += [param["term"]]
                self.fitvalues += [param["val"]]
                self.fiterrors += [param["err"]]

        self.sigma = rmodel.meta["sigma"]
        self.variance = rmodel.meta["variance"]
        self.wssrndf = rmodel.meta["wssrndf"]

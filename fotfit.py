#!/usr/bin/python3
"""
fotfit.py
Photometric response fitter
(c) Martin Jelinek, ASU AV CR, 2023
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.table import Table
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

import termfit

# wishlist :)
# from scipy.interpolate import RegularGridInterpolator

rad = np.pi / 180.0


class FotFit(termfit.termfit):
    """Photometric response fitter"""

    modelname: str = "FotFit photometric model"
    fit_xy: bool = False
    zero: List[float] = []
    px: List[float] = []
    py: List[float] = []
    base_filter: Optional[str] = None
    color_filters: List[str] = []

    def __init__(
        self,
        proj: Optional[str] = None,
        file: Optional[str] = None,
        fit_xy: bool = False,
    ) -> None:
        """Start myself up"""
        super().__init__()
        self.fit_xy = fit_xy
        self.base_filter = None  # Store the filter used for fitting
        self.color_schema = None  # Store filters used for colors

        if file is not None:
            self.readmodel(file)
            self.fixall()

    def __str__(self):
        output = " img --zeropoint--- ------px------ ------py------ \n"

        Nt = len(self.fitterms)

        if self.fit_xy:  # xcoord/ycoord/zeropoint for each image
            N = np.int64((len(self.fitvalues) - len(self.fitterms)) / 3)
        else:
            N = np.int64(len(self.fitvalues) - len(self.fitterms))

        for i in range(0, N):
            try:
                error = self.fiterrors[Nt + i]
            except IndexError:
                error = np.nan
            output += " %03d %6.3f ± %-5.3f " % (
                i,
                self.fitvalues[Nt + i],
                error,
            )  # zero
            if self.fit_xy:
                try:
                    error = self.fiterrors[Nt + N + i]
                except IndexError:
                    error = np.nan
                output += "%6.3f ± %-5.3f " % (self.fitvalues[Nt + N + i], error)  # px
                try:
                    error = self.fiterrors[Nt + 2 * N + i]
                except IndexError:
                    error = np.nan
                output += "%6.3f ± %-5.3f " % (
                    self.fitvalues[Nt + 2 * N + i],
                    error,
                )  # py
            output += "\n"

        output += termfit.termfit.__str__(self)
        return output

    def set_filter_info(self, base_filter: str, color_filters: List[str]) -> None:
        """Store filter information used in the fit"""
        self.base_filter = base_filter
        self.color_schema = color_schema

    def fit(self, data: Tuple[NDArray, ...]) -> OptimizeResult:
        self.fitvalues = self.fitvalues[0 : (len(self.fitterms))] + self.zero
        if self.fit_xy:
            self.fitvalues = np.append(self.fitvalues, np.zeros(2 * len(self.zero)))
        ret = termfit.termfit.fit(self, data)
        return ret

    def zero_val(self) -> Tuple[NDArray, NDArray]:
        p = len(self.fitterms)
        return self.fitvalues[p:], self.fiterrors[p:]

    def mesh_flat(
        self, x: NDArray, y: NDArray, ctrx: float = 512, ctry: float = 512, img: int = 0
    ) -> NDArray:
        """
        Generate flat field data efficiently for a given image using the existing model function.

        Args:
            x: 2D array of x-coordinates
            y: 2D array of y-coordinates
            ctrx: x-coordinate of the image center
            ctry: y-coordinate of the image center
            img: image number

        Returns:
            2D array of flat field data
        """

        x_fine = x.astype(float) + 0.5  # Center of pixels
        y_fine = y.astype(float) + 0.5

        # Normalize coordinates
        coord_x = (x - ctrx) / 1024.0
        coord_y = (y - ctry) / 1024.0

        # Prepare the data array for the model function
        shape = x.shape
        data = (
            np.zeros(shape),  # mc (magnitude)
            np.ones(shape),  # airmass (set to 1 for flat field)
            coord_x,
            coord_y,
            np.zeros(shape),  # color1
            np.zeros(shape),  # color2
            np.zeros(shape),  # color3
            np.zeros(shape),  # color4
            np.full(shape, img),
            np.zeros(shape),  # y (not used in flat field calculation)
            np.ones(shape),  # err (set to 1, not used in model calculation)
            x_fine,
            y_fine,
        )

        return self.model(self.fitvalues, data)

    def old_flat(
        self, x: NDArray, y: NDArray, ctrx: float = 512, ctry: float = 512, img: int = 0
    ) -> NDArray:
        return self.model(
            np.array(self.fitvalues),
            (0, 1, (x - ctrx) / 1024.0, (y - ctry) / 1024.0, 0, 0, 0, 0, img, 0, 0),
        )

    def hyperbola(self, x: float, y: float) -> float:
        return (y * np.sqrt(1.0 * x * x / y / y) + x) / 2.0

    def model(self, values: NDArray, data: Tuple[NDArray, ...]) -> NDArray:
        """Optimized photometric response model"""
        (
            mc,
            airmass,
            coord_x,
            coord_y,
            color1,
            color2,
            color3,
            color4,
            img,
            y,
            err,
            cat_x,
            cat_y,
        ) = data
        values = np.asarray(values)
        img = np.int64(img)

        radius2 = coord_x ** 2 + coord_y ** 2

        # Convert instrumental magnitude to counts
        counts = 10.0 ** (-0.4 * mc)

        # Transform counts to be relative to reference point
        REFERENCE_COUNTS = 10000.0
        OFFSET = 0.01
        transformed_counts = (counts + OFFSET) / REFERENCE_COUNTS
        mct = -2.5 * np.log10(transformed_counts)

        # Calculate base magnitude relative to reference point
        model = mct

        val2 = np.concatenate(
            (values[0 : len(self.fitterms)], np.array(self.fixvalues))
        )

        # Variables to collect rational function parameters
        rational_a = rational_b = rational_c = 0

        for term, value in zip(self.fitterms + self.fixterms, val2):
            if term == "RA":  # Rational function parameter a
                rational_a = value
            elif term == "RB":  # Rational function parameter b
                rational_b = value
            elif term == "RC":  # Rational function parameter c
                rational_c = value
            elif term == "SX":
                # Sinusoidal variation in X direction
                frac_x = cat_x - np.floor(cat_x)
                model += value * np.sin(np.pi * frac_x)
            elif term == "SY":
                # Sinusoidal variation in Y direction
                frac_y = cat_y - np.floor(cat_y)
                model += value * np.sin(np.pi * frac_y)
            elif term == "SXY":
                # Cross-term for X-Y sub-pixel variations
                frac_x = cat_x - np.floor(cat_x)
                frac_y = cat_y - np.floor(cat_y)
                model += value * np.sin(np.pi * frac_x) * np.sin(np.pi * frac_y)
            elif term[0] == "P":
                components = {
                    "A": airmass,
                    "C": color1,
                    "D": color2,
                    "E": color3,
                    "F": color4,
                    "R": radius2,
                    "X": coord_x,
                    "Y": coord_y,
                    "N": mct,
                }
                pterm = value
                n = 1
                for a in term[1:]:
                    if a.isdigit():
                        n = int(a)
                    elif a in components:
                        pterm *= components[a] ** n
                        n = 1
                model += pterm
            elif term == "GA":
                ga = value
            elif term == "GW":
                gw = 10 ** value
            elif term == "EA":
                ea = value
            elif term == "EW":
                ew = value
            elif term == "EQ":
                eq = value
            elif term == "HA":
                ha = value
            elif term == "HM":
                hm = value
            elif term == "HN":
                hn = value
            elif term == "XC":
                bval = np.where(
                    value < 0,
                    value * color1,
                    np.where(value <= 1, value * color2, (value - 1) * color3 + color2),
                )
                model += bval
            elif term == "SC":
                s = np.sin(value * np.pi)
                w1 = 0.5 * (-s + 1)
                w2 = 0.5 * (s + 1)
                bval = np.where(
                    value < -0.5,
                    value * color1,
                    np.where(
                        value < 0.5,
                        w1 * value * color1 + w2 * value * color2,
                        np.where(
                            value <= 1.5,
                            color2
                            + w2 * (value - 1) * color2
                            + w1 * (value - 1) * color3,
                            (value - 1) * color3 + color2,
                        ),
                    ),
                )
                model += bval

        if "ga" in locals() and "gw" in locals():
            model += ga * np.exp(-radius2 * 25 / gw)

        if "ea" in locals() and "ew" in locals():
            foo = np.sqrt(radius2 / ew)
            model += ea * (np.exp(foo) / (foo + 1) - 1)

        if any([rational_a, rational_b, rational_c]):
            x = mct
            rational_correction = (rational_a + rational_b * x) / (1 + rational_c * x)
            model += rational_correction

        if self.fit_xy:
            N = (len(values) - len(self.fitterms)) // 3
            zeros = values[len(self.fitterms) : len(self.fitterms) + N]
            px = values[len(self.fitterms) + N : len(self.fitterms) + 2 * N]
            py = values[len(self.fitterms) + 2 * N : len(self.fitterms) + 3 * N]

            model += px[img] * coord_x + py[img] * coord_y + zeros[img]
        else:
            zeros = values[len(self.fitterms) :]
            model += zeros[img]

        return model

    def isnumber(self, a: str) -> bool:
        try:
            k = int(a)
            return True
        except:
            return False

    def residuals0(self, values: NDArray, data: Tuple[NDArray, ...]) -> NDArray:
        """pure residuals to compute sigma and similar things"""
        (
            mc,
            airmass,
            coord_x,
            coord_y,
            color1,
            color2,
            color3,
            color4,
            img,
            y,
            err,
            cat_x,
            cat_y,
        ) = data
        return np.abs(y - self.model(values, data))

    def residuals(self, values: NDArray, data: Tuple[NDArray, ...]) -> NDArray:
        """residuals for fitting with error weighting and delinearization"""
        (
            mc,
            airmass,
            coord_x,
            coord_y,
            color1,
            color2,
            color3,
            color4,
            img,
            y,
            err,
            cat_x,
            cat_y,
        ) = data
        dist = np.abs((y - self.model(values, data)) / err)
        if self.delin:
            return self.cauchy_delin(dist)
        else:
            return dist

    def oneline(self) -> str:
        """Enhanced model string with filter information"""
        output = []
        # Start with filter information
        if self.base_filter is not None:
            output.append(f"FILTER={self.base_filter}")
        if self.color_schema is not None:
            output.append(f"SCHEMA={self.color_schema}")

        # Add all fitted and fixed terms
        for term, value in zip(
            self.fixterms + self.fitterms, self.fixvalues + self.fitvalues
        ):
            output += f",{term}={value}"

        # For fit_xy mode, explicitly include per-image PX/PY terms
        if self.fit_xy:
            N = (len(self.fitvalues) - len(self.fitterms)) // 3
            px = self.fitvalues[len(self.fitterms) + N : len(self.fitterms) + 2 * N]
            py = self.fitvalues[len(self.fitterms) + 2 * N : len(self.fitterms) + 3 * N]
            for i, (px_val, py_val) in enumerate(zip(px, py)):
                output.append(f"PX{i}={px_val}")
                output.append(f"PY{i}={py_val}")

        return ",".join(output)

    def savemodel(self, file: str) -> None:
        """Enhanced model saving with complete metadata"""
        model_table = Table()

        # Add filter information
        model_table.meta["base_filter"] = self.base_filter
        model_table.meta["color_filters"] = ",".join(self.color_filters)
        model_table.meta["fit_xy"] = self.fit_xy

        # Add terms and values
        terms: List[str] = []
        values: List[float] = []
        errors: List[float] = []

        # Add regular terms
        for term, value in zip(
            self.fitterms + self.fixterms, self.fitvalues + self.fixvalues
        ):
            terms.append(term)
            values.append(value)
            try:
                err = self.fiterrors[terms.index(term)] if term in self.fitterms else 0
            except IndexError:
                err = 0
            errors.append(err)

        # Add per-image PX/PY terms if fit_xy is True
        if self.fit_xy:
            N = (len(self.fitvalues) - len(self.fitterms)) // 3
            px = self.fitvalues[len(self.fitterms) + N : len(self.fitterms) + 2 * N]
            py = self.fitvalues[len(self.fitterms) + 2 * N : len(self.fitterms) + 3 * N]
            for i, (px_val, py_val) in enumerate(zip(px, py)):
                terms.extend([f"PX{i}", f"PY{i}"])
                values.extend([px_val, py_val])
                errors.extend([0, 0])  # These are fitted but we don't save their errors

        model_table["term"] = terms
        model_table["val"] = values
        model_table["err"] = errors

        model_table.meta["name"] = self.modelname
        model_table.meta["sigma"] = self.sigma
        model_table.meta["variance"] = self.variance
        model_table.meta["wssrndf"] = self.wssrndf

        model_table.write(file, format="ascii.ecsv", overwrite=True)

    def readmodel(self, file: str) -> None:
        """Enhanced model reading with filter information"""
        model_table = Table.read(file, format="ascii.ecsv")

        # Read filter information
        self.base_filter = model_table.meta.get("base_filter")
        color_filters = model_table.meta.get("color_filters")
        if color_filters:
            self.color_filters = color_filters.split(",")
        self.fit_xy = model_table.meta.get("fit_xy", False)

        # Clear existing terms
        self.fixterms = []
        self.fixvalues = []
        self.fitterms = []
        self.fitvalues = []
        self.fiterrors = []

        # Read terms, handling PX/PY specially for fit_xy mode
        for param in model_table:
            term = param["term"]
            if term.startswith(("PX", "PY")) and self.fit_xy:
                # These will be handled by the fitting process
                continue
            elif param["err"] == 0:
                self.fixterms.append(term)
                self.fixvalues.append(param["val"])
            else:
                self.fitterms.append(term)
                self.fitvalues.append(param["val"])
                self.fiterrors.append(param["err"])

        # Read statistics
        self.sigma = model_table.meta["sigma"]
        self.variance = model_table.meta["variance"]
        self.wssrndf = model_table.meta["wssrndf"]

    def from_oneline(self, line: str) -> "FotFit":
        """
        Load model from a one-line string representation.

        Args:
            line (str): Model string in format "term1=value1,term2=value2,..."

        Example:
            >>> ffit = fotfit()
            >>> ffit.from_oneline("FILTER=Sloan_r,N1=1.0,PX=0.001,PY=-0.002")
        """
        # Clear existing terms
        self.fixterms = []
        self.fixvalues = []
        self.fitterms = []
        self.fitvalues = []
        self.fiterrors = []

        # Parse terms
        terms = line.split(",")
        img_px = {}  # Store PX terms by image number
        img_py = {}  # Store PY terms by image number

        for term in terms:
            try:
                name, strvalue = term.split("=")
                value = float(strvalue)

                # Handle filter information
                if name == "FILTER":
                    self.base_filter = strvalue
                    continue
                elif name == "COLORS":
                    self.color_filters = strvalue.split(";")
                    continue

                # Handle per-image PX/PY terms
                if name.startswith("PX") and len(name) > 2:
                    img_num = int(name[2:])
                    img_px[img_num] = value
                    continue
                elif name.startswith("PY") and len(name) > 2:
                    img_num = int(name[2:])
                    img_py[img_num] = value
                    continue

                # Add regular term
                self.fixterms.append(name)
                self.fixvalues.append(value)

            except ValueError as e:
                print(f"Warning: Could not parse term '{term}': {e}")
                continue

        # Handle fit_xy mode if per-image PX/PY terms were found
        if img_px or img_py:
            self.fit_xy = True
            max_img = max(
                max(img_px.keys(), default=-1), max(img_py.keys(), default=-1)
            )

            # Fill in any missing image numbers with zeros
            for i in range(max_img + 1):
                if i not in img_px:
                    img_px[i] = 0.0
                if i not in img_py:
                    img_py[i] = 0.0

            # Add sorted PX/PY values to fixvalues
            for i in range(max_img + 1):
                self.fixterms.extend([f"PX{i}", f"PY{i}"])
                self.fixvalues.extend([img_px[i], img_py[i]])

        return self

    def model_from_string(self, model_string: str, data: tuple) -> NDArray:
        """
        Convenience method to load model from string and immediately apply it.

        Args:
            model_string (str): Model definition in one-line format
            data (tuple): Input data tuple for model evaluation

        Returns:
            array-like: Model results

        Example:
            >>> ffit = fotfit()
            >>> result = ffit.model_from_string("N1=1.0,PX=0.001", data_tuple)
        """
        self.from_oneline(model_string)
        return self.model(self.fixvalues, data)

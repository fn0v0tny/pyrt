import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time


class PhotometryData:
    """
    Manages photometric data with complex masking and filtering operations.

    DESIGN PHILOSOPHY:
    PhotometryData handles the complex data management aspects: storage, masking,
    filtering, coordinate transformations, and metadata. It provides methods to
    create immutable FitData snapshots for fitting operations.

    KEY RESPONSIBILITIES:
    - Data storage and organization (_data, _meta dictionaries)
    - Complex masking operations (multiple named masks, mask switching)
    - Data filtering and validation
    - Coordinate transformations and airmass calculations
    - Metadata management

    MASKING WORKFLOW COMPLEXITY:
    The masking system supports non-linear workflows like:
    1. data.use_mask('photometry') → get data → fit
    2. Calculate residuals on ALL data (different mask!)
    3. data.add_mask('combined', new_mask) → switch → refit
    4. Multiple concurrent mask states for different purposes

    SEPARATION OF CONCERNS:
    PhotometryData (data management) ↔ FitData (fitting interface)
    This separation prevents architectural fusion attempts that would create
    subtle bugs due to mask state dependencies and concurrent access patterns.

    WHY NOT FUSION WITH FitData?
    - PhotometryData: Complex, stateful, handles data management
    - FitData: Simple, immutable, handles fitting interface
    - Fusion would create timing bugs with mask switching
    - Current design follows Unix philosophy: "Do one thing well"
    """
    def __init__(self):
        self._data = {}
        self._meta = {}
        self._masks = {}
        self._current_mask = None
        self._filter_columns = []
        self._current_filter = None
        self._required_columns = ["y", "adif", "coord_x", "coord_y", "img", "dy"]
        self._image_counts = {}
        self._total_objects = 0

    def init_column(self, name):
        if name not in self._data and name != "x":
            self._data[name] = []

    def append(self, **kwargs):
        for name, value in kwargs.items():
            if name != "x":
                self.init_column(name)
                self._data[name].append(value)

    def extend(self, **kwargs):
        for name, value in kwargs.items():
            if name != "x":
                self.init_column(name)
                self._data[name].extend(value)
                if name == "img":
                    for img_no in value:
                        self._image_counts[img_no] = (
                            self._image_counts.get(img_no, 0) + 1
                        )

    def set_meta(self, key, value):
        self._meta[key] = value

    def get_meta(self, key, default=None):
        return self._meta.get(key, default)

    def finalize(self):
        for name in self._data:
            self._data[name] = np.array(self._data[name])

        total_objects = sum(self._image_counts.values())
        self.add_mask("default", np.ones(total_objects, dtype=bool))
        self.use_mask("default")

    def add_mask(self, name, mask):
        """Add a new mask, ensuring it matches the data length."""
        if len(mask) != self._total_objects:
            raise ValueError(f"Mask length ({len(mask)}) does not match data length ({self._total_objects})")
        self._masks[name] = mask

    def use_mask(self, name):
        if name not in self._masks:
            raise ValueError(f"Mask '{name}' does not exist.")
        self._current_mask = name

    def get_current_mask(self):
        if self._current_mask is None:
            raise ValueError("No mask is currently active.")
        return self._masks[self._current_mask]

    def compute_colors_and_apply_limits(self, phschema, options):
        """
        Compute colors based on the selected photometric system and apply color limits.

        Args:
        photometric_system (str): Either 'Johnson' or 'AB' to indicate the photometric system.
        options (argparse.Namespace): Command line options containing redlim and bluelim.

        Returns:
        None. Updates the object in-place.
        """
        if photometric_system not in ["Johnson", "AB"]:
            raise ValueError("photometric_system must be either 'Johnson' or 'AB'")

        if photometric_system == "Johnson":
            filters = ["Johnson_B", "Johnson_V", "Johnson_R", "Johnson_I", "J"]
        else:  # AB system
            filters = ["Sloan_g", "Sloan_r", "Sloan_i", "Sloan_z", "J"]

        # Ensure all required filters are present
        for f in filters:
            if f not in self._filter_columns:
                raise ValueError(f"Required filter {f} not found in data")

        # Compute colors
        mags = [self._data[f] for f in filters]
        self._data["color1"] = mags[0] - mags[1]
        self._data["color2"] = mags[1] - mags[2]
        self._data["color3"] = mags[2] - mags[3]
        self._data["color4"] = mags[3] - mags[4]

        # color_mask = self._data['color3'] > 5
        # Apply color limits
        # if options.redlim is not None:
        #    color_mask &= (self._data['color1'] + self._data['color2'])/2 <= options.redlim
        # if options.bluelim is not None:
        #    color_mask &=  (self._data['color1'] + self._data['color2'])/2 >= options.bluelim
        if options.redlim is not None and options.bluelim is not None:
            color_mask = (
                ((self._data["color1"] + self._data["color2"]) / 2 <= options.redlim)
                & ((self._data["color1"] + self._data["color2"]) / 2 >= options.bluelim)
                & (self._data["color3"] > 5)
            )
            self.apply_mask(color_mask)

    def get_arrays(self, *names):
        """Get arrays applying the current mask."""
        self.check_required_columns()

        arrays = []
        current_mask = self._masks[self._current_mask]

        for name in names:
            if name == "x":
                if not self._current_filter:
                    raise ValueError(
                        "No filter is currently set. Use set_current_filter() first."
                    )
                arrays.append(
                    self._data[self._current_filter][self._masks[self._current_mask]]
                )
            elif name in self._data:
                # Ensure data array matches mask length
                if len(self._data[name]) != len(current_mask):
                    raise ValueError(f"Data array '{name}' length ({len(self._data[name])}) "
                                   f"does not match mask length ({len(current_mask)})")
                arrays.append(self._data[name][current_mask])
            else:
                raise KeyError(
                    f"Column '{name}' not found in data. Available columns: {', '.join(self._data.keys())}"
                )
        return tuple(arrays)

    def get_fitdata(self, *names):
        """
        Create an immutable FitData snapshot of the requested data columns.

        This method creates a FitData object containing the current masked state
        of the requested columns. The resulting FitData is independent of this
        PhotometryData object and will not change if masks are switched later.

        Returns:
            FitData: Immutable snapshot with attribute access and tuple compatibility
        """
        arrays = self.get_arrays(*names)
        return FitData(names, *arrays)

    def apply_mask(self, mask, name=None):
        """Apply a new mask, ensuring proper length."""
        if len(mask) != self._total_objects:
            raise ValueError(f"New mask length ({len(mask)}) does not match data length ({self._total_objects})")

        if name is None:
            self._masks[self._current_mask] &= mask
        else:
            self.add_mask(name, self._masks[self._current_mask] & mask)
            self.use_mask(name)

    def reset_mask(self, name=None):
        if name is None:
            name = self._current_mask
        self._masks[name] = np.ones(len(next(iter(self._data.values()))), dtype=bool)

    def add_filter_column(self, filter_name, data):
        if filter_name not in self._data:
            self._data[filter_name] = []
        self._data[filter_name].extend(data)
        if filter_name not in self._filter_columns:
            self._filter_columns.append(filter_name)
        if not self._current_filter:
            self.set_current_filter(filter_name)

    def get_filter_columns(self):
        return self._filter_columns

    def set_current_filter(self, filter_name):
        if filter_name not in self._filter_columns:
            raise ValueError(f"Filter '{filter_name}' not found in data.")
        self._current_filter = filter_name

    def get_current_filter(self):
        return self._current_filter

    def check_required_columns(self):
        missing_columns = [
            col for col in self._required_columns if col not in self._data
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def __len__(self):
        if self._data:
            return np.sum(self._masks[self._current_mask])
        return 0

    def __repr__(self):
        return f"PhotometryData with columns: {list(self._data.keys())}, current filter: {self._current_filter}, current mask: {self._current_mask}"


def make_pairs_to_fit(det, cat, nearest_ind, imgwcs, options, data):
    """
    Efficiently create pairs of data to be fitted.

    :param det: Detection table
    :param cat: Catalog table
    :param nearest_ind: Indices of nearest catalog stars for each detection
    :param imgwcs: WCS object for the image
    :param options: Command line options
    :param data: PhotometryData object to store results
    :return: Number of matched stars added to the data
    """
    try:
        valid_matches = np.array([len(inds) > 0 for inds in nearest_ind])
        det_data = np.array(
            [
                det["X_IMAGE"],
                det["Y_IMAGE"],
                det["MAG_AUTO"],
                det["MAGERR_AUTO"],
                det["ERRX2_IMAGE"],
                det["ERRY2_IMAGE"],
            ]
        ).T[valid_matches]
        cat_inds = np.array([inds[0] if len(inds) > 0 else -1 for inds in nearest_ind])[
            valid_matches
        ]
        cat_data = cat[cat_inds]

        # RA+Dec of detections from their measured X&Y
        ra, dec = imgwcs.all_pix2world(det_data[:, 0], det_data[:, 1], 1)
        # X,Y for catalog for their catalog RA&Dec
        cat_x, cat_y = imgwcs.all_world2pix(cat_data['radeg'], cat_data['decdeg'], 1)

        loc = EarthLocation(
            lat=det.meta["LATITUDE"] * u.deg,
            lon=det.meta["LONGITUD"] * u.deg,
            height=det.meta["ALTITUDE"] * u.m,
        )
        time = Time(det.meta["JD"], format="jd")
        coords = SkyCoord(ra * u.deg, dec * u.deg)
        altaz = coords.transform_to(AltAz(obstime=time, location=loc))
        airmass = altaz.secz.value

        coord_x = (det_data[:, 0] - det.meta["CTRX"]) / 1024
        coord_y = (det_data[:, 1] - det.meta["CTRY"]) / 1024

        filter_mags = {}
        for filter_name in cat.columns:
            if filter_name.startswith(("Sloan_", "Johnson_", "J")):
                filter_mags[filter_name] = cat_data[filter_name]

        magcat = cat_data[det.meta["REFILTER"]]
        mag_mask = (
            (magcat >= options.brightlim) & (magcat <= options.maglim)
            if options.brightlim
            else (magcat <= options.maglim)
        )

        n_matched_stars = np.sum(mag_mask)

        # Store filter data before applying magnitude mask
        for filter_name in cat.filters:
            if filter_name in options.filter_schemas[det.meta['PHSCHEMA']]:
                if filter_name in cat_data.columns:
                    # Apply magnitude mask to get filter data for matched stars only
                    filter_mags[filter_name] = cat_data[filter_name][mag_mask]

        temp_dy = det_data[:, 3][mag_mask]
        temp_dy_no_zero = np.sqrt(np.power(temp_dy, 2) + 0.0004)

        _dx = det_data[:, 4][mag_mask]
        _dy = det_data[:, 5][mag_mask]
        _image_dxy = np.sqrt(np.power(_dx, 2) + np.power(_dy, 2) + 0.0025)

        data.extend(
            y=det_data[:, 2][mag_mask],
            adif=airmass[mag_mask] - det.meta["AIRMASS"],
            coord_x=coord_x[mag_mask],
            coord_y=coord_y[mag_mask],
            img=np.full(n_matched_stars, det.meta["IMGNO"]),
            dy=temp_dy_no_zero,
            image_x=det_data[:, 0][mag_mask],
            image_y=det_data[:, 1][mag_mask],
            image_dxy=_image_dxy,
            ra=cat_data["radeg"][mag_mask],
            dec=cat_data["decdeg"][mag_mask],
        )

        for filter_name, mag_values in filter_mags.items():
            data.add_filter_column(filter_name, mag_values)

        if det.meta["REFILTER"] in filter_mags:
            data.set_current_filter(det.meta["REFILTER"])
        else:
            raise ValueError(
                f"Filter '{det.meta['REFILTER']}' not found in catalog data"
            )

        return n_matched_stars

    except KeyError as e:
        logging.error(f"Error in make_pairs_to_fit: Missing key in detection or catalog data: {e}")
    except ValueError as e:
        logging.error(f"Error in make_pairs_to_fit: Invalid value encountered: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in make_pairs_to_fit: {e}")

    return 0


def compute_initial_zeropoints(data, metadata):
    """
    Compute initial zeropoints for each image based on the selected best filter.

    Args:
    data (PhotometryData): Object containing all photometry data.
    metadata (list): List of metadata for each image.

    Returns:
    list: Initial zeropoints for each image.
    """
    zeropoints = []
    x, y, img = data.get_arrays("x", "y", "img")

    for img_meta in metadata:
        img_mask = img == img_meta["IMGNO"]
        img_x = x[img_mask]
        img_y = y[img_mask]

        if len(img_x) > 0:
            zeropoint = np.median(img_x - img_y)  # x (catalog mag) - y (observed mag)
        else:
            logging.warning(
                f"No data for image {img_meta['IMGNO']}, using default zeropoint of 0"
            )
            zeropoint = 0

        zeropoints.append(zeropoint)

    return zeropoints

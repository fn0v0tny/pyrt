"""Run the transient processing pipeline on a set of detections."""
from catalog import QueryParams, setup_catalog_cache
from transients import *
from transient_analyser import *
from extraction_manager import ImageExtractionManager

import os

data_dir = "./data/test15/"
sample_image = "./im30-0.ecsv"
cache_dir = "./catalog_cache"
setup_catalog_cache(cache_dir)
detection_table = []
for file in os.listdir(data_dir):
    if file.endswith(".ecsv"):
        print(file)
        detection_table.append(open_ecsv_file(data_dir + file))
print(detection_table)
image_manager = ImageExtractionManager(detection_table)
ra, dec = image_manager.field_center
image_manager.generate_images()
first_det = open_ecsv_file(data_dir + sample_image, verbose=True)
query_params = QueryParams(**{
        "ra": ra,
        "dec": dec,
        "width": 1.2 * first_det.meta["FIELD"],
        "height": 1.2 * first_det.meta["FIELD"],
        "mlim": 20,
    }
)
transient_analyzer = TransientAnalyzer()
multi_analyzer = MultiDetectionAnalyzer(transient_analyzer)

reliable_candidates, lightcurves = multi_analyzer.process_detection_tables_with_lightcurves(
    detection_tables=detection_table,
    catalogs=["atlas@local", "gaia", "usno"],
    params=query_params,
    idlimit=3.0,
    radius_check=20.0,
    filter_pattern="r",
    min_n_detections=5,
    min_catalogs=3,
    min_quality=0.1,
)
print(reliable_candidates)
reliable_candidates.write("candidates_test15.tbl", format="ascii.ipac", overwrite=True)

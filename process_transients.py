"""Run the transient processing pipeline on a set of detections."""
from catalog import QueryParams
from transients import *
from transient_analyser import *

import os
data_dir = "./data/test4/"
sample_image = "20240414022243-676-clear-012-df.ecsv"
detection_table = []
for file in os.listdir(data_dir):
    if "clear" in file:
        detection_table.append(open_ecsv_file(data_dir+file))
first_det = open_ecsv_file(data_dir+sample_image, verbose=True)
query_params = QueryParams(**{"ra": first_det.meta["CTRRA"],
    "dec": first_det.meta["CTRDEC"],
    "width": 1.2*first_det.meta["FIELD"],
    "height": 1.2*first_det.meta["FIELD"],
    "mlim": 20})
transient_analyzer = TransientAnalyzer()
multi_analyzer = MultiDetectionAnalyzer(
    transient_analyzer
)

reliable_candidates = multi_analyzer.process_detection_tables(
    detection_tables=detection_table,
    catalogs=['atlas@local', 'gaia', 'usno'],
    params=query_params,
    idlimit=2.0,
    radius_check=20.0,
    filter_pattern='r',
    min_n_detections=5,
    min_catalogs=3,
    min_quality=0.05,
)
print(reliable_candidates)
reliable_candidates.write("candidates_test3.tbl", format="ascii.ipac", overwrite=True)

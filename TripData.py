import pandas
from panas_evaluation import PanasEvaluation
from ecg import EcgAnalyzer

DELIMETER = ";"


class TripData:
    def __init__(self, trip_id, filename):
        self.trip_id = trip_id
        self.filename = filename
        self.pre_specific = pandas.read_csv(f"./data/{filename}/PRE_SPECIFIC.csv", delimiter=DELIMETER)
        self.pre_demographic = pandas.read_csv(f"./data/{filename}/PRE_DEMOGRAPHIC.csv", delimiter=DELIMETER)
        self.pre_panas_data = pandas.read_csv(f"./data/{filename}/PRE_TRIP_PANAS.csv", delimiter=DELIMETER)
        self.post_panas_data = pandas.read_csv(f"./data/{filename}/POST_TRIP_PANAS.csv", delimiter=DELIMETER)
        self.post_specific = pandas.read_csv(f"./data/{filename}/POST_SPECIFIC.csv", delimiter=DELIMETER)
        self.calibration_data = pandas.read_csv(f"./data/{filename}/trip_{trip_id}_calibration_data.csv",
                                                delimiter=DELIMETER)
        self.ecg_calibration_data = pandas.read_csv(f"./data/{filename}/trip_{trip_id}_ecg_calibration_data.csv",
                                                    delimiter=DELIMETER)
        self.data = pandas.read_csv(f"./data/{filename}/trip_{trip_id}_data.csv", delimiter=DELIMETER)
        self.ecg_data = pandas.read_csv(f"./data/{filename}/trip_{trip_id}_ecg_data.csv", delimiter=DELIMETER)

        #PANAS
        self.pre_panas_evaluated = PanasEvaluation(self.pre_panas_data.values)
        self.post_panas_evaluated = PanasEvaluation(self.post_panas_data.values)

    def get_pre_panas_evaluated(self):
        return self.pre_panas_evaluated

    def get_post_panas_evaluated(self):
        return self.post_panas_evaluated

    def get_calibration_ecg_data_evaluated(self):
        analyzer = EcgAnalyzer(self.ecg_calibration_data.ecg_mv)
        return analyzer.process()

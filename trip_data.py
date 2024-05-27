import pandas
from panas_evaluation import PanasEvaluation
from ecg import EcgAnalyzer
from other.linear_acceleration import LinearAccelerationAnalyzer
from other.data_plotter import DataPlotter

DELIMETER = ";"


class TripData:
    def __init__(self, trip_id, filename, path):
        self.trip_id = trip_id
        self.filename = filename
        self.pre_specific = pandas.read_csv(f"{path}/{filename}/PRE_SPECIFIC.csv", delimiter=DELIMETER)
        self.pre_demographic = pandas.read_csv(f"{path}/{filename}/PRE_DEMOGRAPHIC.csv", delimiter=DELIMETER)
        self.pre_panas_data = pandas.read_csv(f"{path}/{filename}/PRE_TRIP_PANAS.csv", delimiter=DELIMETER)
        self.pre_mssq_1 = pandas.read_csv(f"{path}/{filename}/PRE_MSSQ_1.csv", delimiter=DELIMETER)
        self.pre_mssq_2 = pandas.read_csv(f"{path}/{filename}/PRE_MSSQ_2.csv", delimiter=DELIMETER)
        self.pre_bsss = pandas.read_csv(f"{path}/{filename}/PRE_BSSS.csv", delimiter=DELIMETER)
        self.post_panas_data = pandas.read_csv(f"{path}/{filename}/POST_TRIP_PANAS.csv", delimiter=DELIMETER)
        self.post_specific = pandas.read_csv(f"{path}/{filename}/POST_SPECIFIC.csv", delimiter=DELIMETER)
        self.calibration_data = pandas.read_csv(f"{path}/{filename}/trip_{trip_id}_calibration_data.csv",
                                                delimiter=DELIMETER)
        self.ecg_calibration_data = pandas.read_csv(f"{path}/{filename}/trip_{trip_id}_ecg_calibration_data.csv",
                                                    delimiter=DELIMETER)
        self.data = pandas.read_csv(f"{path}/{filename}/trip_{trip_id}_data.csv", delimiter=DELIMETER)
        self.ecg_data = pandas.read_csv(f"{path}/{filename}/trip_{trip_id}_ecg_data.csv", delimiter=DELIMETER)

        # PANAS
        self.pre_panas_evaluated = PanasEvaluation(self.pre_panas_data.values)
        self.post_panas_evaluated = PanasEvaluation(self.post_panas_data.values)

    def get_trip_id(self):
        return self.trip_id

    def get_data(self):
        return self.data

    def get_comfort_score(self):
        return self.post_specific["answer"][0]

    def evaluate_linear_acceleration(self):
        linear_acc = LinearAccelerationAnalyzer(self.calibration_data, self.data)
        return linear_acc

    def plot(self):
        # analyzer = EcgAnalyzer(self.ecg_calibration_data.ecg_mv)
        # analyzer.process()
        data_plotter = DataPlotter(self.calibration_data, self.data)
        data_plotter.plot()

    def get_pre_panas_evaluated(self):
        return self.pre_panas_evaluated

    def get_post_panas_evaluated(self):
        return self.post_panas_evaluated

    def get_calibration_ecg_data_evaluated(self):
        analyzer = EcgAnalyzer(self.ecg_calibration_data.ecg_mv)
        return analyzer.process()

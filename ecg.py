# Taken from https://github.com/paulvangentcom/heartrate_analysis_python/blob/master/examples/2_regular_ECG/Analysing_a_regular_ECG_signal.ipynb

import heartpy as hp
import matplotlib.pyplot as plt
from scipy.signal import resample

RESAMPLE_SCALE = 6
ECG_SAMPLE_RATE = 140


class EcgAnalyzer:
    def __init__(self, data):
        self.data = data

    def process(self, show_plot=True):
        filtered = hp.filter_signal(self.data, cutoff=0.05, sample_rate=ECG_SAMPLE_RATE, filtertype='notch')

        # resample the data. Usually 2, 4, or 6 times is enough depending on original sampling rate
        resampled_data = resample(filtered, len(filtered) * RESAMPLE_SCALE)

        # And run the analysis again. Don't forget to up the sample rate as well!
        measured_data, measures = hp.process(hp.scale_data(resampled_data), ECG_SAMPLE_RATE * RESAMPLE_SCALE)

        if show_plot:
            hp.plotter(measured_data, measures)
            plt.show()
            # display computed measures
            for measure in measures.keys():
                print('%s: %f' % (measure, measures[measure]))

        return measured_data, measures

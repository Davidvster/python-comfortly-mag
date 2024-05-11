import matplotlib.pyplot as plt
import pandas


class LinearAccelerationAnalyzer:
    def __init__(self, calibration_data, trip_data):
        self.trip_data = trip_data
        self.calibration_data = calibration_data

    def process(self, show_plot=True):
        trimmed_calibration_data = self.calibration_data.iloc[50:-50]
        calibration_x_avg = trimmed_calibration_data['linear_accelerometer_x_axis_linear_acceleration'].mean()
        calibration_y_avg = trimmed_calibration_data['linear_accelerometer_y_axis_linear_acceleration'].mean()
        calibration_z_avg = trimmed_calibration_data['linear_accelerometer_z_axis_linear_acceleration'].mean()

        # actual_new_x = self.trip_data['linear_accelerometer_x_axis_linear_acceleration'] - calibration_x_avg
        # actual_new_y = self.trip_data['linear_accelerometer_y_axis_linear_acceleration'] - calibration_y_avg
        # actual_new_z = self.trip_data['linear_accelerometer_z_axis_linear_acceleration'] - calibration_z_avg

        actual_new_x = self.trip_data['linear_accelerometer_x_axis_linear_acceleration']
        actual_new_y = self.trip_data['linear_accelerometer_y_axis_linear_acceleration']
        actual_new_z = self.trip_data['linear_accelerometer_z_axis_linear_acceleration']

        if show_plot:
            # plt.plot(actual_new_x[:5000], label='X')
            # plt.plot(actual_new_y[:20000], label='Y')
            # plt.plot(actual_new_z[:5000], label='z')
            # plt.plot(self.trip_data['gps_speed'][:20000] / 10, label='speed')

            plt.plot(actual_new_x, label='X')
            plt.plot(actual_new_y, label='Y')
            # plt.plot(actual_new_z, label='z')
            plt.plot(self.trip_data['gps_speed'] / 10, label='speed')

            plt.xlabel('Time')
            plt.legend()
            plt.show()

        return actual_new_x, actual_new_y, actual_new_z

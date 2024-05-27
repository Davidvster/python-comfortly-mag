import matplotlib.pyplot as plt
import pandas


class DataPlotter:
    def __init__(self, calibration_data, trip_data):
        self.trip_data = trip_data
        self.calibration_data = calibration_data

    def plot(self):
        trimmed_trip_data = self.trip_data.iloc[50:-50]

        fig, axs = plt.subplots(4, 1, figsize=(20, 16))
        axs[0].plot(trimmed_trip_data['accelerometer_x_axis_acceleration'], label='X')
        axs[0].plot(trimmed_trip_data['accelerometer_y_axis_acceleration'], label='Y')
        axs[0].plot(trimmed_trip_data['accelerometer_z_axis_acceleration'], label='Z')
        axs[0].set_title('Accelerometer')
        axs[0].legend()

        axs[1].plot(trimmed_trip_data['gravity_x_axis_gravity'], label='X')
        axs[1].plot(trimmed_trip_data['gravity_y_axis_gravity'], label='Y')
        axs[1].plot(trimmed_trip_data['gravity_z_axis_gravity'], label='Z')
        axs[1].set_title('Gyroscope')
        axs[1].legend()

        axs[2].plot(trimmed_trip_data['heart_rate_bpm'], label='BPM')
        axs[2].set_title('Heart rate (BPM)')
        axs[2].legend()

        axs[3].plot(trimmed_trip_data['gps_speed'], label='m/s')
        axs[3].set_title('GPS speed')
        axs[3].legend()

        plt.savefig('foo.png')
        plt.tight_layout()
        plt.show()


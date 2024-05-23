from TripData import TripData
from ecg import EcgAnalyzer

trip_id = "32"
filename = trip_id + "_mesto_16-05-2024_20-41"

# izracunaj PANAS score pre and post
# average values
# sum sprememb
# stevilo spike-ov
# segmentacija kjer so spike-ti
# Calculate parameters from ECG -> RR interval, QT, PR, etc. https://litfl.com/qt-interval-ecg-library/ https://github.com/tejasa97/ECG-Signal-Processing

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = TripData(trip_id, filename)
    # ecg = data.get_calibration_ecg_data_evaluated()
    # linearAcceleration = data.evaluate_linear_acceleration().process(True)
    data.plot()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# To join two arrays in Python with timestamps and a 50ms threshold, you can use the following steps:
#
# First, you'll need to import the datetime module to be able to work with timestamps.
# Then, you can iterate over the elements in the first array, and for each element, find the corresponding element in the second array that is within the 50ms threshold of the timestamp of the first element. To do this, you can use a loop and compare the timestamps of the elements using the timedelta object from the datetime module.
# When you find a matching element in the second array, you can add both elements to a new list or array, which will be your merged array.
# You can continue this process until you have checked all the elements in the first array, and then return the merged array as the result.
# Here's an example of how you could implement this in Python:

# from datetime import datetime, timedelta
#
# def merge_arrays(arr1, arr2):
#     merged_arr = []
#     for elem1 in arr1:
#         timestamp1 = elem1[0]
#         for elem2 in arr2:
#             timestamp2 = elem2[0]
#             if timestamp2 - timestamp1 < timedelta(milliseconds=50):
#                 merged_arr.append((timestamp1, elem1[1], elem2[1]))
#                 break
#     return merged_arr
#
# arr1 = [(datetime(2022, 1, 1, 12, 0, 0, 0), "elem1"), (datetime(2022, 1, 1, 12, 0, 0, 25), "elem2"), (datetime(2022, 1, 1, 12, 0, 0, 75), "elem3")]
# arr2 = [(datetime(2022, 1, 1, 12, 0, 0, 25), "elem4"), (datetime(2022, 1, 1, 12, 0, 0, 50), "elem5"), (datetime(2022, 1, 1, 12, 0, 1, 0), "elem6")]
#
# merged_arr = merge_arrays(arr1, arr2)
# print(merged_arr)


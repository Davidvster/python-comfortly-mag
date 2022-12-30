import pandas
from panas_evaluation import PanasEvaluation

DELIMETER = ";"

trip_id = "57"
filename = trip_id + "_Ajdovščina 26. 12_26-12-2022_19-19"

# izracunaj PANAS score pre and post
# average values
# sum sprememb
# stevilo spike-ov
# segmentacija kjer so spike-ti


def read_data():
    pre_specific = pandas.read_csv(f"./data/{filename}/PRE_SPECIFIC.csv", delimiter=DELIMETER)
    pre_demographic = pandas.read_csv(f"./data/{filename}/PRE_DEMOGRAPHIC.csv", delimiter=DELIMETER)
    pre_panas = pandas.read_csv(f"./data/{filename}/PRE_TRIP_PANAS.csv", delimiter=DELIMETER)
    post_panas = pandas.read_csv(f"./data/{filename}/POST_TRIP_PANAS.csv", delimiter=DELIMETER)
    post_specific = pandas.read_csv(f"./data/{filename}/POST_SPECIFIC.csv", delimiter=DELIMETER)
    calibration_data = pandas.read_csv(f"./data/{filename}/trip_{trip_id}_calibration_data.csv", delimiter=DELIMETER)
    ecg_calibration_data = pandas.read_csv(f"./data/{filename}/trip_{trip_id}_ecg_calibration_data.csv", delimiter=DELIMETER)
    data = pandas.read_csv(f"./data/{filename}/trip_{trip_id}_data.csv", delimiter=DELIMETER)
    ecg_data = pandas.read_csv(f"./data/{filename}/trip_{trip_id}_ecg_data.csv", delimiter=DELIMETER)

    pre_panas = PanasEvaluation(pre_panas.values)
    post_panas = PanasEvaluation(post_panas.values)
    print(pre_panas.pa, pre_panas.na)
    print(post_panas.pa, post_panas.na)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pa_items = ["interested", "excited", "strong", "determined", "attentive", "alert", "enthusiastic", "inspired",
                "active",
                "proud"]
    string = "1. Zainteresirano (Interested)"

    if any(s in string.lower() for s in pa_items):
        print("The string contains at least one of the strings in the array")
    else:
        print("The string does not contain any of the strings in the array")
    read_data()

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


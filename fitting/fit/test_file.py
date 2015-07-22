import numpy as np;

def helper_splitting_array(pos_array, neg_array, step_size, step_size_factor):
    # Point of this is because the pos_Array is changing
    # Need a copy to add the numbers properly
    pos_copy = np.copy(pos_array)
    neg_copy = np.copy(neg_array)

    for index in range(0, len(pos_array)):
        next_number = pos_copy[index] + step_size
        if next_number > 0 and next_number not in pos_array:
            pos_array = np.sort(np.insert(pos_array, index + 1, next_number))

    #Point of second loop is in case it goes towards negatives
    # Farnaz, Redundancy of same numbers
    for index in range(0, len(neg_array)):
        next_number = neg_copy[index] - step_size

        if next_number > 0 and next_number not in neg_array:
            neg_array = np.sort(np.insert(neg_array, index + 1, next_number))

    step_size = step_size * step_size_factor
    return(pos_array, neg_array, step_size)

a = np.array([0.5, 1.0, 2.0, 3])
print(helper_splitting_array(a, a, 0.5, 0.98))
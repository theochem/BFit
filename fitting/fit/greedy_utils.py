



class GreedyMBIS():
    def __init__(self):
        pass


def check_redundancies(coeffs, exps):
    new_coeffs = coeffs.copy()
    new_exps = exps.copy()

    indexes_where_they_are_same = []
    for i, alpha in enumerate(exps):
        similar_indexes = []
        for j in range(i + 1, len(exps)):
            if j not in similar_indexes:
                if np.abs(alpha - exps[j]) < 1e-2:
                    if i not in similar_indexes:
                        similar_indexes.append(i)
                    similar_indexes.append(j)
        if len(similar_indexes) != 0:
            indexes_where_they_are_same.append(similar_indexes)

    for group_of_similar_items in indexes_where_they_are_same:
        for i in  range(1, len(group_of_similar_items)):
            new_coeffs[group_of_similar_items[0]] += coeffs[group_of_similar_items[i]]

    if len(indexes_where_they_are_same) != 0:
        print("-------- Redundancies found ---------")
        print()
        new_exps = np.delete(new_exps, [y for x in indexes_where_they_are_same for y in x[1:]])
        new_coeffs = np.delete(new_coeffs, [y for x in indexes_where_they_are_same for y in x[1:]])
    assert len(exps) == len(coeffs)
    return new_coeffs, new_exps


def get_next_possible_coeffs_and_exps(factor, coeffs, exps):
    size = exps.shape[0]
    all_choices_of_exponents = []
    all_choices_of_coeffs = []
    coeff_value = 100.
    for index, exp in np.ndenumerate(exps):
        if index[0] == 0:
            exponent_array = np.insert(exps, index, exp / factor)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        elif index[0] <= size:
            exponent_array = np.insert(exps, index, (exps[index[0] - 1] + exps[index[0]]) / 2)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        all_choices_of_exponents.append(exponent_array)
        all_choices_of_coeffs.append(coefficient_array)
        if index[0] == size - 1:
            exponent_array = np.append(exps, np.array([exp * factor]))
            all_choices_of_exponents.append(exponent_array)
            all_choices_of_coeffs.append(np.append(coeffs, np.array([coeff_value])))
    return all_choices_of_coeffs, all_choices_of_exponents
def round_probabilities(lst):
    delta = 1e-2
    # round items in list if just above 1 or just below 0 due to float issues
    for i in range(len(lst)):
        if lst[i] > 1:
            if lst[i]-1 < delta:
                lst[i] = 1
            else:
                print(i, lst[i])
                raise Exception("probability > 1 for item {}: {}".format(i, lst[i]))
        if lst[i] < 0:
            if -lst[i] < delta:
                lst[i] = 0
            else:
                raise Exception("probability < 0 for item {}: {}".format(i, lst[i]))
    return lst

def roundl(lst, precision):
    return [round(x, precision) for x in lst]
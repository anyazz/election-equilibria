import numpy as np
import scipy.io
import random
import json

random.seed(2020)
np.set_printoptions(threshold=np.inf)

# networks: list of dicts for each community network, comprising following columns
def main():
    cols = ('relationsMatrix', 'grade', 'race', 'scode', 'sex', 'totalnoms')

    for i in range(10, 85):
        print("loading " + str(i))
        file_dict = scipy.io.loadmat('data/matlab_matrices/comm' + str(i) + '.mat')
        dct = {k: file_dict[k] for k in cols}
        # assert check_noms(i, dct['relationsMatrix'], dct['totalnoms'])
        dct['trustMatrix'] = calculate_influence(dct['relationsMatrix'])
        
        with open('data/json/comm' + str(i) + '.json', 'w') as f:
            data = json.dumps(dct, cls=NumpyEncoder)
            f.write(data)


# sanity check: ensure that number of outgoing edges <= total nominations for each node
def check_noms(i, relationsMatrix, totalnoms):
    for j, row in enumerate(relationsMatrix):
        count = sum([1 if x > 0 else 0 for x in row])
        if count > totalnoms[j]:
            print("ERROR IN FILE " + str(i) + ", ITEM " + str(j))
            return False
    return True

# calculate row-stochastic trust matrix given relations matrix
# uses random value for self-trust
def calculate_influence(relationsMatrix):
    trust_matrix = []
    for j, row in enumerate(relationsMatrix):
        row_sum = sum(row)
        if row_sum:
            self_confidence = random.random()
            trust_row = [x * (1-self_confidence)/row_sum for x in row]
            trust_row[j] = self_confidence
        else:
            trust_row = [0] * len(row)
            trust_row[j] = 1
        trust_matrix.append(trust_row)
    return trust_matrix

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

main()
from ADR.preprocess import *

class ADR():
    def __init__(self, hdegree=1, add_parity=False, add_tf=False):
        self.hdegree = hdegree
        self.add_parity = add_parity
        self.add_tf = add_tf

    def fit(self, x_train, y_train=None):
        if y_train is not None:
            x_train, _ = split_to_N_AN(x_train, y_train)
        x_train = random_samples(x_train, min(x_train.shape[1]*10, x_train.shape[0]))
        extended_x_train = extend_2darray(x_train, self.hdegree, self.add_parity, self.add_tf)
        self.ns_x_train = scipy.linalg.null_space(extended_x_train)

        # if x_train.shape[0] > 10000:
        #     sample_x_train = x_train[0:10000, :]
        #     remain_x_train = x_train[10000:, :]
        #
        #     ns_sample_x_train = scipy.linalg.null_space(sample_x_train)
        #     dot_Remain_nsSample = np.dot(remain_x_train, ns_sample_x_train).__abs__()
        #     idx_valid = ((dot_Remain_nsSample > 1e-10).sum(axis=0)) < 1
        #     self.ns_x_train = ns_sample_x_train[:, idx_valid]
        # else:
        #     self.ns_x_train = scipy.linalg.null_space(x_train)

        # print(f'ns shape is {self.ns_x_train.shape}')
        return self.ns_x_train.shape[1]

    def pred(self, x):
        extended_x = extend_2darray(x, self.hdegree, self.add_parity, self.add_tf)
        y_pred = (np.dot(extended_x, self.ns_x_train).__abs__() > 1e-10).sum(axis=1) > 0
        return y_pred

    def evaluate(self, x, y):
        y_pred = self.pred(x)
        precision, recall, F, mcc = p_r_f_mcc(y_pred, y)
        return precision, recall, F, mcc

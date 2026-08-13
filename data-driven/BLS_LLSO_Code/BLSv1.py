import numpy as np
from sklearn.preprocessing import OneHotEncoder
#from numpy import linalg as LA

def tansig(x):
    return (2 / (1 + np.exp(-2*x))) - 1
def pinv(A, C):
    return np.linalg.inv(A.T.dot(A) + C * np.eye(A.shape[1])).dot(A.T)
def sparse_bls(A, B, lam=0.001, itrs=50):
    AA = A.T.dot(A)
    m, n = A.shape[1], B.shape[1]
    ok = np.zeros((m, n))
    uk = np.zeros((m, n))
    L1 = np.linalg.inv(AA + np.eye(m))
    L2 = L1.dot(A.T).dot(B)
    for _ in range(itrs):
        ck = L2 + L1.dot(ok - uk)
        ok = np.sign(ck + uk) * np.maximum(np.abs(ck + uk) - lam, 0)
        uk += ck - ok
    return ok

class BLSClassifier:
    def __init__(self, C=0.001, NumFea=30, NumWin=6, NumEnhan=60):
        self.C = C#正则化系数
        self.NumFea = NumFea
        self.NumWin = NumWin
        self.NumEnhan = NumEnhan
        self.WF = []#fea map随机权重
        self.WFSparse = []# 稀疏映射权重
        self.WeightEnhan = None# enhance层权重
        self.WeightTop = None# 输出层权重
        self.means = []
        self.ranges = []

    def fit(self, X, y):
        #one hot转换
        enc = OneHotEncoder(sparse_output=False)
        Y_onehot = enc.fit_transform(y.reshape(-1,1))
        N, d = X.shape

        #生成fea map和enhance的随机权重
        for i in range(self.NumWin):
            w = 2 * np.random.randn(d+1, self.NumFea) - 1
            self.WF.append(w)
        self.WeightEnhan = 2 * np.random.randn(self.NumWin*self.NumFea+1, self.NumEnhan) - 1
        #fea map
        H1 = np.hstack([X, 0.1*np.ones((N,1))])
        Ymap = np.zeros((N, self.NumWin*self.NumFea))

        #稀疏
        for i in range(self.NumWin):
            A1 = H1.dot(self.WF[i])
            # 归一化至±1
            A1 = 2*(A1 - A1.min(axis=0))/(np.ptp(A1,axis=0)) - 1
            Ws = sparse_bls(A1, H1).T
            self.WFSparse.append(Ws)
            T1 = H1.dot(Ws)
            #记录归一化参数
            mu, rng = T1.mean(axis=0), np.ptp(T1,axis=0)
            self.means.append(mu); self.ranges.append(rng)
            Ymap[:, i*self.NumFea:(i+1)*self.NumFea] = (T1 - mu) / rng
        #enhance
        H2 = np.hstack([Ymap, 0.1*np.ones((N,1))])
        T2 = tansig(H2.dot(self.WeightEnhan))
        #输出层权重求解
        T3 = np.hstack([Ymap, T2])
        self.WeightTop = pinv(T3, self.C).dot(Y_onehot)
        return self

    def predict(self, X):
        N = X.shape[0]
        H1 = np.hstack([X, 0.1*np.ones((N,1))])
        Ymap = np.zeros((N, self.NumWin*self.NumFea))
        for i in range(self.NumWin):
            T1 = H1.dot(self.WFSparse[i])
            mu, rng = self.means[i], self.ranges[i]
            Ymap[:, i*self.NumFea:(i+1)*self.NumFea] = (T1 - mu) / rng
        H2 = np.hstack([Ymap, 0.1*np.ones((N,1))])
        T2 = tansig(H2.dot(self.WeightEnhan))
        T3 = np.hstack([Ymap, T2])
        #选最大的输出
        scores = T3.dot(self.WeightTop)
        return np.argmax(scores, axis=1)

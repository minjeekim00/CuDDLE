import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm
​
​
​
class Barcode():
    def __init__(self, real_latent, fake_latent, outlier_prob=0, explainability=1, steps=100, plot_step=1e-4):
        self.outlier_prob = outlier_prob
        self.explain = explainability
        self.real_latent = real_latent
        self.fake_latent = fake_latent
        self.steps = steps
        self.plot_step = plot_step
        assert [len(self.real_latent.shape), len(self.fake_latent.shape)] == [2,2], print("Latent dimension should be 2: (number of latent vectors, dimension of latent vectors)")
    
    def svd(self, x):
        _, num = x.shape
        num = int(num * self.explain)
        u, s, vh = np.linalg.svd(x)
        smat = np.zeros((u.shape[0], s.shape[0]))
        smat[:s.shape[0], :s.shape[0]] = np.diag(s)
        return u, smat, vh[:,:num]
    
    def get_distance(self):
        print("Calculating distances for every combination ...")
        if self.explain<1:
            ur, sr, vhr = self.svd(self.real_latent)
            uf, sf, vhf = self.svd(self.fake_latent)
​
            reduction_r = np.dot(np.dot(ur, sr), vhr)
            reduction_f = np.dot(np.dot(uf, sf), vhf)
        else:
            reduction_r = self.real_latent
            reduction_f = self.fake_latent
        
        rnum, _ = reduction_r.shape
        fnum, _ = reduction_f.shape
        dists = []
        
        for i in tqdm.tqdm(range(rnum)):
            for j in range(fnum):
                dists.append(np.sqrt(np.mean((reduction_r[i]-reduction_f[j])**2)))
        dists = sorted(dists)
        dists = np.array(dists)
        dists = dists[:int(len(dists)*(1-self.outlier_prob))]
        return dists / (dists.max()+1e-4)
    
    def get_barcode(self):
        fidelity = 0
        dists = self.get_distance()
        interval = 1 / self.steps
        bars = []
        for i in range(self.steps):
            b = np.sum(dists<interval*i)
            bars.append(b)
        bars = np.array(bars)
        if np.max(bars)!=0:
            bars = bars / bars.max()
        diversity = bars.std()
        for i in range(self.steps):
            x=np.arange(0,bars[i],self.plot_step)
            plt.plot(x, [i/self.steps]*len(x), 'b-')
            ith_score = bars[i] * (1/self.steps)
            fidelity += ith_score
        plt.ylim(0,1.1)
        plt.savefig('./barcode.png')
        plt.show()
        return 1-fidelity, 1.5 * diversity

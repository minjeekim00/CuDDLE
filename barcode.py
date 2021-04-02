import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt
import math
from sklearn.metrics import pairwise_distances
import time
import multiprocessing
from multiprocessing import Pool

class Barcode():
    def __init__(self, real_latent, fake_latent, distance=2, outlier_prob=0, outlier_position=None, explainability=1, steps=100, plot_step=1e-4):
        self.outlier_prob = outlier_prob
        self.explain = explainability
        self.real_latent = real_latent
        self.fake_latent = fake_latent
        self.steps = steps
        self.plot_step = plot_step
        self.outlier_position = outlier_position
        self.distance = distance
     
        self.dist_metric = None
        self.dists = {'rr':None, 'rf':None, 'ff':None}
        assert self.outlier_position in ['in', 'out', 'both', None]
        assert [len(self.real_latent.shape), len(self.fake_latent.shape)] == [2,2], print("Latent dimension should be 2: (number of latent vectors, dimension of latent vectors)")
        assert isinstance(self.distance, int)

        if self.distance == 2:
            self.dist_metric = 'l2'
            self.dist_fx = self.L2
        else:
            self.dist_fx = self.Lp
    
    def L2(self, x,y):
        return np.sqrt(np.sum((x-y)**2))

    def Lp(self, x,y):
        return np.linalg.norm(x-y, ord=self.distance)
        
    def compare_dicts(self, real, fake):
        if len(real) != len(fake):
            return 0
        else:
            for i in range(len(real)):
                if (real[i] == fake[i]).all():
                    pass
                else:
                    return 0
            return 1

    def svd(self, x):
        _, num = x.shape
        u, s, vh = np.linalg.svd(x)
        for i in range(len(s)):
            if np.sum(s[:i])>=(self.explainability*np.sum(s)):
                num=i
                break
        smat = np.zeros((u.shape[0], s.shape[0]))
        smat[:s.shape[0], :s.shape[0]] = np.diag(s)
        return u, smat, vh[:,:num]

    def compute_pairwise_distance(self, data_x, data_y):
        dists = []

        flag = self.compare_dicts(data_x, data_y)
        
        if flag:
            for i in tqdm.tqdm(range(len(data_x)), position=0):
                for j in range(len(data_y)):
                    if i != j:
                        dists.append(self.dist_fx(data_x[i], data_y[j]))
        else:
            if self.distance == 2:
                dists = pairwise_distances(data_x, data_y, metric=self.dist_metric)
            else:
                for i in tqdm.tqdm(range(len(data_x)), position=0):
                    for j in range(len(data_y)):
                        dists.append(self.dist_fx(data_x[i], data_y[j]))
        return np.array(dists).reshape((-1))

    def compute_distance(self, multi, real, fake):
        print("Calculating distances for every combination ...")
        if self.explain<1:
            ur, sr, vhr = self.svd(real)
            uf, sf, vhf = self.svd(fake)
            
            reduction_r = np.dot(np.dot(ur, sr), vhr)
            reduction_f = np.dot(np.dot(uf, sf), vhf)
        else:
            reduction_r = real
            reduction_f = fake
        
        rnum, _ = reduction_r.shape
        fnum, _ = reduction_f.shape
        
        dists = self.compute_pairwise_distance(reduction_r, reduction_f)
        
        start = time.time()
        if multi:
            dists = self.parallel_sort(dists)
        else:
            dists = self.sort(dists)
        print(f"Sorting distance took {(time.time()-start)/60} mins.....")
        print(f"Calculating diversity for {len(dists)} combinations ...")
        
        if self.outlier_position=='in':
            dists = dists[int(len(dists)*self.outlier_prob):]
        elif self.outlier_position=='out':
            dists = dists[:int(len(dists)*(1-self.outlier_prob))]
        elif self.outlier_position=='both':
            dists = dists[int(len(dists)*self.outlier_prob):int(len(dists)*(1-self.outlier_prob))]
        
        return dists
     
    def sort(self, dists):
        dists = np.array(sorted(dists))
        return dists
    
    def parallel_sort(self, dists):
        processes = multiprocessing.cpu_count()
        pool = Pool(processes=processes)
        size = int(math.ceil(float(len(dists)) / processes))
        dists = [dists[i * size:(i + 1) * size] for i in range(processes)]
        dists = pool.map(self.sort, dists)
        pool.close()
        pool.join()
        return np.hstack(dists)
    
    def get_distance_norm(self, dists):
        dists = dists / (dists.max()+1e-4)
        return dists
    
    def get_diversity(self, dists):
        dists = self.get_distance_norm(dists)
        diversity = dists.std()
        return diversity
    
    def get_fidelity(self, dists):
        dists = self.get_distance_norm(dists)
        interval = 1 / self.steps
        bars = self.get_bars(dists)
        fidelity = np.mean(bars) # the area of cdf-like curve equals the mean
        return fidelity
    
    def get_bars(self, dists):
        interval = 1 / self.steps
        bars = []
        for i in range(self.steps):
            b = np.sum(dists<interval*i)
            bars.append(b)
        bars = np.array(bars)
        if np.max(bars)!=0:
            bars = bars / bars.max()
        return bars

    def plot_bars(self, bars, title='Barcode', filename='./barcode.png', format=None):
        
        print("Plotting fidelity for {} samples ...".format(len(bars)))
        for i in range(self.steps):
            x=np.arange(0,bars[i],self.plot_step)
            plt.plot(x, [i/self.steps]*len(x), 'b-')
        plt.ylim(0,1.1)
        plt.title(title)
        plt.savefig(filename, format=format)
        plt.show()
        plt.close('all')
        
    def get_barcode(self, multi=True):
        
        if self.dists['rr'] is None:
            print("Distance not found. Computing distances between Real and Real")
            self.dists['rr'] = self.compute_distance(multi, self.real_latent, self.real_latent)
        if self.dists['rf'] is None:
            print("Distance not found. Computing distances between Real and Fake")    
            realfake = self.compute_distance(multi, self.real_latent, self.fake_latent)
        if self.dists['ff'] is None:
            print("Distance not found. Computing distances between Fake and Fake")    
            fakefake = self.compute_distance(multi, self.fake_latent, self.fake_latent)
        

        rf_fidelity = self.get_fidelity(self.dists['rf'])
        rr_fidelity = self.get_fidelity(self.dists['rr'])
        ff_fidelity = self.get_fidelity(self.dists['ff'])

        rf_diversity = self.get_diversity(self.dists['rf'])
        rr_diversity = self.get_diversity(self.dists['rr'])
        ff_diversity = self.get_diversity(self.dists['ff'])

        print(f"Real vs Fake Fidelity : {rf_fidelity:.3f} | Real vs Real Fidelity : {rr_fidelity:.3f} | Fake vs Fake Fidelity : {ff_fidelity:.3f}")
        print(f"Real vs Fake Diversity: {rf_diversity:.3f} | Real vs Real Diversity: {rr_diversity:.3f} | Fake vs Fake Diversity: {ff_diversity:.3f}")

        return {"mutual_fidelity": rf_fidelity, "relative_fidelity": rf_fidelity/rr_fidelity,\
                "mutual_diversity":rf_diversity,"relative_diversity":rf_diversity/(np.sqrt(rr_diversity) * np.sqrt(ff_diversity))}

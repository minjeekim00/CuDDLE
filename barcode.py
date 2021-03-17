import numpy as np
import multiprocessing
import sklearn.metrics
import math

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
        self.dists = None
        self.bars = None
        assert self.outlier_position in ['in', 'out', 'both', None]
        assert [len(self.real_latent.shape), len(self.fake_latent.shape)] == [2,2], print("Latent dimension should be 2: (number of latent vectors, dimension of latent vectors)")
        
        if self.distance == 2:
            self.dist_metric = 'l2'

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

    def compute_pairwise_distance(self, data_x, data_y=None):
        import sklearn.metrics
        
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(
            data_x, data_y, metric=self.dist_metric)
        return dists

    def compute_distance(self, multiprocessing):
        import time
        if self.explain<1:
            ur, sr, vhr = self.svd(self.real_latent)
            uf, sf, vhf = self.svd(self.fake_latent)
            
            reduction_r = np.dot(np.dot(ur, sr), vhr)
            reduction_f = np.dot(np.dot(uf, sf), vhf)
        else:
            reduction_r = self.real_latent
            reduction_f = self.fake_latent
        
        rnum, _ = reduction_r.shape
        fnum, _ = reduction_f.shape
        
        dists = self.compute_pairwise_distance(reduction_r, reduction_f)
        dists = np.reshape(dists, (-1))
        
        start = time.time()
        if multiprocessing:
            dists = self.parallel_sort(dists)
        else:
            dists = self.sort(dists)
        print(f"Sorting distance took {(time.time()-start)/60} mins.....")
        print("Calculating diversity for {} combinations ...".format(len(dists)))
        
        if self.outlier_position=='in':
            dists = dists[int(len(dists)*self.outlier_prob):]
        elif self.outlier_position=='out':
            dists = dists[:int(len(dists)*(1-self.outlier_prob))]
        elif self.outlier_position=='both':
            dists = dists[int(len(dists)*self.outlier_prob):int(len(dists)*(1-self.outlier_prob))]
        
        self.dists = dists
     
    def sort(self, dists):
        return np.array(sorted(dists))
    
    def parallel_sort(self, dists):
        from multiprocessing import Pool
        processes = multiprocessing.cpu_count()
        pool = Pool(processes=processes)
        size = int(math.ceil(float(len(dists)) / processes))
        dists = [dists[i * size:(i + 1) * size] for i in range(processes)]
        dists = pool.map(self.sort, dists)
        pool.close()
        pool.join()
        return np.hstack(dists)
        
    def get_distance(self):
        return self.dists
    
    def get_distance_norm(self):
        dists = self.dists
        dists = dists / (dists.max()+1e-4)
        return dists
    
    def get_diversity(self):
        dists = self.get_distance()
        diversity = dists.std()
        return diversity
    
    def get_fidelity(self):
        dists = self.get_distance_norm()
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
        self.bars = bars
        return bars

    def plot_bars(self, title='Barcode', filename='./barcode.png', format=None):
        import matplotlib.pyplot as plt
        assert self.bars is not None
        
        print("Plotting fidelity for {} samples ...".format(len(self.bars)))
        for i in range(self.steps):
            x=np.arange(0,self.bars[i],self.plot_step)
            plt.plot(x, [i/self.steps]*len(x), 'b-')
        plt.ylim(0,1.1)
        plt.title(title)
        plt.savefig(filename, format=format)
        plt.show()
        plt.close('all')
        
    def get_barcode(self, multiprocessing=True):
        if self.dists is None:
            print("Distance not found. Computing...")
            self.compute_distance(multiprocessing)
        diversity = self.get_diversity()
        fidelity = self.get_fidelity()
        return fidelity, diversity

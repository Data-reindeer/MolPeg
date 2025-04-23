import math
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['MolPeg']


def cal_scores(outputs, method = 'confidence'):    
    if method == 'confidence':                 
        p = torch.sigmoid(outputs)
        p[p<0.5] = 1 - p[p<0.5]                  
        score = p

    elif method == 'entropy':
        p = torch.sigmoid(outputs)
        entropy = (-p * torch.log(p + 1e-6) - (1-p) * torch.log(1 - p + 1e-6))
        score = entropy
    return score

class PreBatch(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        return data, int(index)

class MolPeg(Dataset):
    def __init__(self, dataset, ratio = 0.5, num_epoch=None, delta = 0.875, scores = None, method='molpeg'):
        super().__init__()
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.scores = np.ones(len(self.dataset)) if scores is None else scores
        self.transform = dataset.transform
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0
        self.method = method

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        weight = float(self.weights[index])
        # return data, torch.tensor(index), torch.tensor(weight)
        return data, int(index), weight
    
    def __setscore__(self, indices, values):
        # if (self.scores[indices] != values).any(): pdb.set_trace()
        self.scores[indices] = values

    def prune(self, seed):
        if self.method in ['molpeg', 'rankloss']:
            tup = [(idx, score) for idx, score in enumerate(self.scores)]
            tup = sorted(tup, key=lambda x:x[1], reverse=True)
            # tup = sorted(tup, key=lambda x:x[1])
            keep_samples = np.array([idx for idx, _ in tup])[:int(self.ratio * len(self.dataset))]
            self.reset_weights()
            if len(keep_samples)>0:
                # Tips: ratio is keep ratio, rather than pruning ratio
                self.weights[keep_samples] = 1/self.ratio
            print('Keep {} samples for next iteration'.format(len(keep_samples)))
            self.save_num += len(keep_samples)
            np.random.shuffle(keep_samples)

        elif self.method in ['curloss']: 
            func = 'logistic'
            def logistic(x, center, k):
                return 1 / (1 + math.exp(-k * (x - center)))
            tup = [(idx, score) for idx, score in enumerate(self.scores)]
            tup = sorted(tup, key=lambda x:x[1], reverse=True)
            # Define hard and easy ratio for pruning
            if func == 'linear': 
                p = 1 - (seed / self.num_epoch)
            elif func == 'logistic':
                p = 1 - logistic(seed, 50, 1)
                print(p)
            hard_num = int(self.ratio * p * len(self.dataset))
            easy_num = int(self.ratio * (1-p) * len(self.dataset))
            hard_idxs = [idx for idx, _ in tup][:hard_num]
            easy_idxs = [idx for idx, _ in tup][-easy_num:] if easy_num >= 1 else []
            keep_samples = np.array(hard_idxs+easy_idxs)
            
            assert math.isclose(keep_samples.shape[0] / len(self.dataset), self.ratio, abs_tol=0.001)
            self.reset_weights()
            if len(keep_samples)>0:
                # Tips: ratio is keep ratio, rather than pruning ratio
                self.weights[keep_samples] = 1/self.ratio
            print('Keep {} samples for next iteration'.format(len(keep_samples)))
            self.save_num += len(keep_samples)
            np.random.shuffle(keep_samples)


        elif self.method == 'random':
            keep_samples  = np.random.randint(0, self.__len__(), int(self.ratio*self.__len__()))
            print('Keep {} samples for next iteration'.format(len(keep_samples)))
            self.save_num += len(keep_samples)
            np.random.shuffle(keep_samples)            
        
        return keep_samples

    def pruning_sampler(self):
        return MolPegSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return MolPegSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))

    def init_scores(self, scores):
        self.scores = scores

class MolPegSampler():
    def __init__(self, molpeg_dataset, num_epoch = math.inf, delta = 1):
        self.molpeg_dataset = molpeg_dataset
        
        self.seq = None
        self.stop_prune = num_epoch
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.molpeg_dataset.reset_weights()
            self.seq = self.molpeg_dataset.no_prune()
        else:
            self.seq = self.molpeg_dataset.prune(self.seed)
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)
        

    def __next__(self):
        
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self
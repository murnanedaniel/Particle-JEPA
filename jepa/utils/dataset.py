import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import numpy as np
from toytrack import ParticleGun, Detector, EventGenerator
from typing import Optional, Union, List

class TracksDataset(IterableDataset):
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            minbias_lambda: Optional[float] = 50,
            pileup_lambda: Optional[float] = 45,
            hard_proc_lambda: Optional[float] = 5,
            minbias_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            pileup_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            hard_proc_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
            warmup_t0: Optional[float] = 0
        ):
        super().__init__()

        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.minbias_lambda = minbias_lambda
        self.pileup_lambda = pileup_lambda
        self.hard_proc_lambda = hard_proc_lambda
        self.minbias_pt_dist = minbias_pt_dist
        self.pileup_pt_dist = pileup_pt_dist
        self.hard_proc_pt_dist = hard_proc_pt_dist

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        return _TrackIterable(
            self.hole_inefficiency,
            self.d0,
            self.noise,
            self.minbias_lambda,
            self.pileup_lambda,
            self.hard_proc_lambda,
            self.minbias_pt_dist,
            self.pileup_pt_dist,
            self.hard_proc_pt_dist,
        )
    
class _TrackIterable:
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            minbias_lambda: Optional[float] = 50,
            pileup_lambda: Optional[float] = 45,
            hard_proc_lambda: Optional[float] = 5,
            minbias_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            pileup_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            hard_proc_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
            warmup_t0: Optional[float] = 0
        ):
        
        detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel', 
            min_radius=0.5, 
            max_radius=3, 
            number_of_layers=10,
        )
        
        self.minbias_gun = ParticleGun(
            dimension=2, 
            num_particles=[minbias_lambda, None, "poisson"], 
            pt=minbias_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.pileup_gun = ParticleGun(
            dimension=2, 
            num_particles=[pileup_lambda, None, "poisson"],
            pt=pileup_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.hard_proc_gun = ParticleGun(
            dimension=2, 
            num_particles=[hard_proc_lambda, None, "poisson"],
            pt=hard_proc_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.minbias_gen = EventGenerator(self.minbias_gun, detector, noise)
        self.hard_proc_gen = EventGenerator([self.pileup_gun, self.hard_proc_gun], detector, noise)
        
        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.minbias_lambda = minbias_lambda
        self.pileup_lambda = pileup_lambda
        self.hard_proc_lambda = hard_proc_lambda
        self.minbias_pt_dist = minbias_pt_dist
        self.pileup_pt_dist = pileup_pt_dist
        self.hard_proc_pt_dist = hard_proc_pt_dist
    
    def __next__(self):
        
        y = (np.random.rand() < 0.5)
        
        if y:
            event = self.hard_proc_gen.generate_event()
        else:
            event = self.minbias_gen.generate_event()
        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        mask = torch.ones(x.shape[0], dtype=bool)

        return x, mask, torch.tensor([y], dtype=torch.float), event.particles
    
def collate_fn(ls):
    x, mask, y, events = zip(*ls)
    return pad_sequence(x, batch_first=True), pad_sequence(mask, batch_first=True), torch.cat(y), list(events)

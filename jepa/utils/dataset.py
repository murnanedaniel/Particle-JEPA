import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import numpy as np
from toytrack import ParticleGun, Detector, EventGenerator
from typing import Optional, Union, List

class TracksDatasetVariable(IterableDataset):
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
            min_radius: Optional[float] = 0.5,
            max_radius: Optional[float] = 3.,
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
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __iter__(self):
        return _TrackIterableVariable(
            self.hole_inefficiency,
            self.d0,
            self.noise,
            self.minbias_lambda,
            self.pileup_lambda,
            self.hard_proc_lambda,
            self.minbias_pt_dist,
            self.pileup_pt_dist,
            self.hard_proc_pt_dist,
            self.min_radius,
            self.max_radius
        )
    
class _TrackIterableVariable:
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
            min_radius: Optional[float] = 0.5,
            max_radius: Optional[float] = 3.,
        ):
        
        detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel', 
            min_radius=min_radius, 
            max_radius=max_radius, 
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
        self.y = np.random.rand() < 0.5
    
    def __next__(self):
        
        self.y = not self.y
        
        if self.y:
            event = self.hard_proc_gen.generate_event()
        else:
            event = self.minbias_gen.generate_event()
            
        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        mask = torch.ones(x.shape[0], dtype=bool)

        return x, mask, torch.tensor([self.y], dtype=torch.float), event
    
def collate_fn_variable(ls):
    x, mask, y, events = zip(*ls)
    return pad_sequence(x, batch_first=True), pad_sequence(mask, batch_first=True), torch.cat(y), list(events)


class TracksDatasetFixed(IterableDataset):
    
    def __init__(
        self,
        hole_inefficiency: Optional[float] = 0,
        d0: Optional[float] = 0.1,
        noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
        num_particles: Optional[float] = 5,
        pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
        min_radius: Optional[float] = 0.5,
        max_radius: Optional[float] = 3.,
    ):
        super().__init__()

        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.num_particles = num_particles
        self.pt_dist = pt_dist
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __iter__(self):
        return _TrackIterableFixed(
            self.hole_inefficiency,
            self.d0,
            self.noise,
            self.num_particles,
            self.pt_dist,
            self.min_radius,
            self.max_radius
        )
    
class _TrackIterableFixed:
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            num_particles: Optional[float] = 5,
            pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
            min_radius: Optional[float] = 0.5,
            max_radius: Optional[float] = 3.,
        ):

        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.num_particles = num_particles
        self.pt_dist = pt_dist
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel', 
            min_radius=min_radius, 
            max_radius=max_radius, 
            number_of_layers=8,
        )

        self.particle_gun = ParticleGun(
            dimension=2, 
            pt=pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )
        self.particle_gen = EventGenerator(particle_gun = self.particle_gun, 
                                           detector = detector, 
                                           num_particles = self.num_particles, 
                                           noise = self.noise)
        
        

    def __next__(self):
        
        event = self.particle_gen.generate_event()
            
        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        mask = torch.ones(x.shape[0], dtype=bool)

        return x, mask, event

def collate_fn_fixed(ls):
    x, mask, events = zip(*ls)
    return pad_sequence(x, batch_first=True), pad_sequence(mask, batch_first=True), list(events)
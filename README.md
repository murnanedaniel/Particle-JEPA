## Particle-JEPA

### TODO

- [ ] Understand this current repository
- [ ] Add in the v1 implementation of Particle-JEPA training task

#### v1 Implementation
- [ ] Implement patchifier: voxel / binning in 3D / 2D
- [ ] Implement patch encoder: 
- [ ] Implement context/target sampler
- [ ] Implement predictor

To install:
```
git clone https://github.com/ryanliu30/gtrack.git --recurse-submodules
cd gtrack
conda create -n gtrack python=3.10
conda activate gtrack
pip install -r requirements.txt
pip install -e .
pip install -e ToyTrack
```

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
git clone https://github.com/murnanedaniel/Particle-JEPA.git --recurse-submodules
cd Particle-JEPA
conda create -n jepa python=3.10
conda activate jepa
pip install -r requirements.txt
pip install -e .
```

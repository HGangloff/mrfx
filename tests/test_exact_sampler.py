import os

os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["JAX_ENABLE_X64"] = "1"
import mrfx

potts = mrfx.models.Potts(beta=0.1, K=2)
es = mrfx.samplers.ExactSampler(lx=4, ly=4)  # max before memory overflow: (22, 22)
es.z_recursion(potts)

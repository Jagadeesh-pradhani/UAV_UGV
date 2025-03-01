import faiss
import numpy as np

d = 64                           # Dimension
nb = 100000                      # Database size
nq = 10000                       # Number of queries
np.random.seed(1234)             # Make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index = faiss.IndexFlatL2(d)
index.add(xb)
D, I = index.search(xq, 5)

print("First 5 results:", I[:5])
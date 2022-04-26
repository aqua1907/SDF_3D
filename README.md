# Researching
Papers:
 - [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/pdf/2006.09661v1.pdf)
 - [DeepSDF]([https://arxiv.org/pdf/1901.05103.pdf])
 - [DIST](https://arxiv.org/pdf/1911.13225.pdf)
 - [SDF-SRN](https://arxiv.org/pdf/2010.10505.pdf)
 - [Acorn](https://arxiv.org/pdf/2105.02788.pdf)
 - [SDFDiff](https://arxiv.org/pdf/1912.07109.pdf)
 - [Gradient-SDF](https://arxiv.org/pdf/2111.13652.pdf)
 - [StyleSDF](https://arxiv.org/pdf/2112.11427.pdf)

Useful links:
 - [Signed Distance Function Representation, Tracking, and Mapping](https://courses.cs.washington.edu/courses/cse571/16au/slides/10-sdf.pdf)
 - https://github.com/fogleman/sdf
 - https://iquilezles.org/articles/distfunctions2d/
 - https://github.com/marian42/mesh_to_sdf
 - https://github.com/mikedh/trimesh
 - https://github.com/lucidrains/siren-pytorch

Videos:
 - [CSC2547 DeepSDF](https://www.youtube.com/watch?v=1iuLxJmQII0)
 - [The SDF of a Box](https://www.youtube.com/watch?v=62-pRVZuS5c&t=1s)
 - [Understanding the SDF](https://www.youtube.com/watch?v=QgzxBN1m9WE&t=834s)
 - [SDF (signed distance field)](https://www.youtube.com/watch?v=ca2g4K5cxKY)
 - [ACORN](https://www.youtube.com/watch?v=P192X3J6cg4&t=88s)

# Results | Benchmarks
|    | F1-Score | Object size| Batch size | Mean batch time (ms) | Mean sample time  (us) | Number of parameters |
| ------------- | ----------------|--------------|--------------------|------------------------|----------------------|---
| **Chair** | 0.886 | 0.63 MB | 16384  | 0.40 ms| 0.02 us | 330K |
| **Handgun** | 0.901 | 0.31 MB | 1024 | 0.38 ms | 0.38 us | 330K | 
| **Pixar-lamp** | 0.877 | 0.12 MB |  16384 | 0.38 ms  | 0.02 us| 330K |
| **Plane** | 0.918 | 1.53 MB | 16384 | 0.39 | 0.02 us | 330K |
| **Raven** | 0.704 | 2.53 MB | 16384 | 0.39 | 0.02 us | 330K |
| **Spaceship** | 0.918 MB | 8.73 MB | 16384 | 0.39 ns | 0.02 us | 330K |
| **Sword** | 0.733 | 0.17 MB | 10240 | 0.40 ms | 0.04 us | 330K |
| **Coffee-table** | 0.835 MB | 0.18 | 32768 | 0.40 ms  | 0.01 us| 330K |


#Overview
1) Used 8 different meshes. The most simple (Pixar-lamp) has 5,3K vertices. The most complex (Spaceship) has (330K)
2) Used Siren network from this [paper](https://arxiv.org/pdf/2006.09661v1.pdf)
3) Correct combination of LR and batch size is individual for each input mesh
4) L1 Loss better converges rather than L2 Loss
5) Larger scan resolution and number of points of input mesh can obtain higher F1-Score
6) The best combination for SIREN is 6 hidden layers with 256 neurons each
7) "Raven" mesh has the lowest F1-Score. I guess this is because of the overlapping feathers on the bird's wings, hence it is more complicated to compute SDF

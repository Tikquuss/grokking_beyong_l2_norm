This is the code to reproduce our ICML2025 paper titled **Grokking Beyond the Euclidean Norm of Model Parameters** (https://arxiv.org/abs/2506.05718).

Below, you will find the notebook to use to reproduce each figure.
* For compressed sensing and matrix factorization, the experiments do not use GPUs; CPUs are enough: the routines are implemented with `numpy`.
* For the other experiments (modular addition with MLP, non-linear teacher-student, Sobolev training), a single GPU (specifically, the T4 GPU from Google Colab) is sufficient.

The notebooks are included in the code.

| Figures | Notebook                |
|---------|-------------------------|
|    1    | algorithmic_dataset_MLP.ipynb|
|    2    | algorithmic_dataset_MLP.ipynb|
|    3    | compressed_sensing.ipynb|
|    4    | compressed_sensing.ipynb|
|    5    | matrix_factorization.ipynb|
|    6    | matrix_factorization.ipynb|
|    7    | compressed_sensing.ipynb|
|    8    | matrix_factorization.ipynb|
|    9    | compressed_sensing.ipynb|
|   10    | non_linear_teacher_student.ipynb|
|   11    | non_linear_teacher_student_sobolev.ipynb|
|   12    | compressed_sensing.ipynb|
|   13    | compressed_sensing.ipynb|
|   14    | matrix_factorization.ipynb|
|   15    | matrix_factorization.ipynb|
|   16    | compressed_sensing.ipynb|
|   17    | compressed_sensing.ipynb|
|   18    | compressed_sensing.ipynb|
|   19    | compressed_sensing.ipynb|
|   20    | compressed_sensing.ipynb|
|   21    | compressed_sensing.ipynb|
|   22    | compressed_sensing.ipynb|
|   23    | matrix_factorization.ipynb|
|   24    | compressed_sensing.ipynb|
|   25    | compressed_sensing.ipynb|
|   26    | compressed_sensing.ipynb|
|   27    | compressed_sensing.ipynb|
|   28    | compressed_sensing.ipynb|
|   29    | matrix_factorization.ipynb|
|   30    | algorithmic_dataset_MLP.ipynb|
|   31    | non_linear_teacher_student.ipynb|
|   32    | non_linear_teacher_student.ipynb|
|   33    | non_linear_teacher_student.ipynb|
|   34    | non_linear_teacher_student_sobolev.ipynb|

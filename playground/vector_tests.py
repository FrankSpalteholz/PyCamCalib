import numpy as np



list = [[-8], [4]]
vec = np.array(list)

vec_norm = np.linalg.norm(vec)
vec_norm_sqrt = pow(vec_norm,2)

print(vec_norm)



print(vec_norm_sqrt)
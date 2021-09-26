import numpy as np
import argparse

parser = argparse.ArgumentParser(description='compute various properties of \
                                 two categorical distribution')
parser.add_argument('-distrA',  type=float, help="first distribution",
                    nargs='+')
parser.add_argument('-distrB',  type=float, help="second distribution",
                    nargs='+')

args = parser.parse_args()
categorical_A = args.distrA
categorical_B = args.distrB

assert isinstance(categorical_A, list)
assert isinstance(categorical_B, list)

A = np.zeros((4, 4))
B = np.zeros((4, 4))

#categorical_A = [0.25, 0.05, 0.2, 0.5]

for row, p_i in enumerate(categorical_A):
    for col, p_j in enumerate(categorical_A):
        if row == col:
            A[row, col] = p_i*(1 -p_i)
        else:
            A[row, col] = -p_i*p_j

#categorical_B = [0.3, 0.1, 0.15, 0.45]

for row, p_i in enumerate(categorical_B):
    for col, p_j in enumerate(categorical_B):
        if row == col:
            B[row, col] = p_i*(1 -p_i)
        else:
            B[row, col] = -p_i*p_j

print('\nDistribution A:\n{}'.format(categorical_A))
print('Distribution B:\n{}'.format(categorical_B))

print("\nCovarianza A:")
print(A)
print("Covarianza B:")
print(B)

eigA, eigVectA = np.linalg.eig(A)
eigB, eigVectB = np.linalg.eig(B)

print("\nAutovettori A:")
print(eigVectA)
print("Autovettori B:")
print(eigVectB)

print('\nProdotti scalari:')
dot_prod = np.zeros(len(categorical_A))
for i in range(4):
    dot_prod[i] = np.dot(eigVectA[:, i], B[:, i])
    print('{}: {}'.format(i, dot_prod[i]))

mean_dot = np.mean(dot_prod)
print('\nMedia prodotti scalari: {}'.format(mean_dot))

entropyA = -sum([p*np.log(p) for p in categorical_A])
entropyB = -sum([p*np.log(p) for p in categorical_B])

print('\nEntropia')
print('distr A: {}'.format(entropyA))
print('distr B: {}'.format(entropyB))

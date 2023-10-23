# If you need 2 * k - th order of approximation set M = k

# for rational (no more than 5)
# M = 3
# A = reshape([(j - 1 // 2)^(2 * i - 1) for j in 1:M for i in 1:M], M, M)
# b = [i != 1 ? 0 : 1 // 2 for i in 1:M]

# coef = A \ b

# @show coef

# for any 
M = 2
A = reshape([(j - 1 / 2)^(2 * i - 1) for j in 1:M for i in 1:M], M, M)
b = [i != 1 ? 0 : 1 / 2 for i in 1:M]

@show A \ b
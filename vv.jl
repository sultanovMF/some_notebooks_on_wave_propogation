using Base.Threads;
using BenchmarkTools;

N = 100
a = zeros(N)

@threads for i = 1:N
  a[i] = i
end

print(a)
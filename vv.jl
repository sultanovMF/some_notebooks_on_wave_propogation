using Plots;
using Base.Threads;
using LoopVectorization;
using ProgressBars;



Threads.nthreads() = 8

const NX = 100
const NY = 100
z = zeros(NX, NY)



@tturbo for i in 1:NX
  for j in 1:NY
    z[i, j] = i + j
  end
end
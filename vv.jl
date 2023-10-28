# using Plots;
# using Base.Threads;
# using LoopVectorization;
# using ProgressBars;

# Threads.nthreads() = 8

# NX = 10
# NY = 10

# z = zeros(NX, NY)

# Nt = 10

# for n in 1:Nt
#   @tturbo for i in 1:NX
#     for j in 1:NY
#        z[i, j] = z[i, j] + 1
#     end
#   end
# end

# @show z

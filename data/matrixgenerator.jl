using LinearAlgebra
using SparseArrays

# define f(x, y)
function f(x, y, k)
    return 2k*sin.(π*x)*sin(π*y')
end


# define the exact solution
function exact(x,y)
    return sin.(π*x)*sin.(π*y')
end



# generate A matrix, write to file
k = 2 # thermal diffusivity constant
Δx = 0.025
x = 0:Δx:1
Δy = Δx
y = 0:Δy:1
N = length(y)-1 # N+1 total nodes in x and y direction
println("Total Nodes: ", N+1)
println("Interior Nodes: ", N-1)

Ix = sparse(I, N-1, N-1)
Iy = sparse(I, N-1, N-1)

D2 = sparse(1:N-1, 1:N-1, -2*ones(N-1)) +
        sparse(2:N-1,1:N-2,ones(N-2),N-1,N-1) +
        sparse(1:N-2,2:N-1,ones(N-2),N-1,N-1)
Dxx = kron(D2, Iy)
Dyy = kron(Ix, D2)

Dxx = (k / Δx^2) * Dxx
Dyy = (k / Δy^2) * Dyy

A = Dxx + Dyy

A_io = open("data/A.txt", "w")
println(A_io, (N-1)^2)
println(A_io, A[:])
close(A_io)


# generate B, write to file
#N = 10
B = zeros(N-1,N-1)

for i = 1:N-1
    for j = 1:N-1
        B[i,j] = f(i,j,k)
    end
end

B = B[:]

b_io = open("data/b.txt", "w")
println(b_io, (N-1)^2^2)
for i = 1:(N-1)^2
    println(b_io, B[i])
end
close(b_io)


# generate exact solution, write to file
#N = 10
ANS = zeros(N-1,N-1)

for i = 1:N-1
    for j = 1:N-1
        ANS[i,j] = exact(i,j)
    end
end

ANS .= ANS[:]

ans_io = open("data/ans.txt", "w")
println(ans_io, (N-1)^2)
for i = 1:(N-1)^2
    println(ans_io, ANS[i])
end
close(ans_io)

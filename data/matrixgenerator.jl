using LinearAlgebra
using SparseArrays

# define f(x, y)
function f(x, y)
    return -6
end


# define the exact solution
function exact(x,y)
    return 1 + x^2 + 2*y^2
end


# generate A matrix, write to file
k = 1
T = 1
λ = 0.1
Δx = 0.05
x = 0:Δx:1
Δy = Δx
y = 0:Δy:1
N = length(y)-1 # N+1 total nodes in x and y direction

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
A = A[:]

A_io = open("data/A.txt", "w")
println(A_io, (N-1)^2)
for i = 1:N^2
    println(A_io, A[i])
end
close(A_io)




# generate B, write to file
#N = 100
B = zeros(N-1,N-1)

for i = 1:N-1
    for j = 1:N-1
        B[i,j] = f(i,j)
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
ANS = zeros(N-1,N-1)

for i = 1:N-1
    for j = 1:N-1
        ANS[i,j] = exact(i,j)
    end
end

ANS = ANS[:]

ans_io = open("data/ans.txt", "w")
println(ans_io, (N-1)^2)
for i = 1:(N-1)^2
    println(ans_io, ANS[i])
end
close(ans_io)


using LinearAlgebra

# define f(x, y)
function f(x, y)
    return -6
end


# define the exact solution
function exact(x,y)
    return 1 + x^2 + 2*y^2
end


# generate B, write to file
N = 1000
B = zeros(N,N)[:]

b_io = open("b.txt", "w")
println(ans_io, length(B))
println(b_io, B)
close(b_io)


# generate exact solution, write to file
N = 1000
ANS = exact(N,N)

for i = 1:N
    for j = 1:N
        ANS[i,j] = exact(i,j)
    end
end

ANS = ANS[:]

ans_io = open("ans.txt", "w")
println(ans_io, length(ANS))
println(ans_io, ANS)
close(ans_io)


# generate A matrix, write to file
k = 1
T = 1
λ = 0.1
Δx = 0.025
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

A_io = open("A.txt", "w")
println(A_io, (N-1)^2)
println(A_io, A[:])
close(A_io)

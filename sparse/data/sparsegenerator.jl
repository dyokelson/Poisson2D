using LinearAlgebra
using SparseArrays

# define F(x, y)
function F(x, y, k)
    return 2*k*π^2*sin.(π*x)*sin(π*y')
end


# define the exact solution
function exact(x,y)
    return sin.(π*x)*sin.(π*y')
end



# generate A matrix, write to file
k = 2 # thermal diffusivity constant
Δx = 0.001
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

Dxx = (-k / Δx^2) * Dxx
Dyy = (-k / Δy^2) * Dyy

A = Dxx + Dyy
A = sparse(A')

A_io = open("A_001.txt", "w")
println(A_io, A.m, " ", A.n, " ", last(A.colptr) - 1)

for i = 1:length(A.nzval)
    if i <= length(A.colptr)
        println(A_io, round(A.nzval[i], digits=3), " ", A.rowval[i] - 1, " ", A.colptr[i] - 1)
    end

    if i > length(A.colptr)
        println(A_io, round(A.nzval[i], digits=3), " ", A.rowval[i] - 1)
    end
end
close(A_io)

# generate B, write to file
B = zeros(N-1,N-1)

for i = 1:N-1
    for j = 1:N-1
        B[i,j] = F(x[i+1],y[j+1],k)
    end
end

B = B[:]

println(B[1])
b_io = open("b_001.txt", "w")
println(b_io, (N-1)^2)
for i = 1:(N-1)^2
    println(b_io, B[i])
end
close(b_io)

# generate exact solution, write to file
ANS = zeros(N-1,N-1)

for i = 1:N-1
    for j = 1:N-1
        ANS[i,j] = exact(x[i+1],y[j+1])
    end
end

ANS = ANS[:]

ans_io = open("ans_001.txt", "w")
println(ans_io, (N-1)^2)
for i = 1:(N-1)^2
    println(ans_io, ANS[i])
end
close(ans_io)



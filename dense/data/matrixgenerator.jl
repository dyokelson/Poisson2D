using LinearAlgebra
using SparseArrays
using Plots
using IterativeSolvers

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

Dxx = (-k / Δx^2) * Dxx
Dyy = (-k / Δy^2) * Dyy

A = Dxx + Dyy
A_Dense = Matrix(A)

A_io = open("A_025.txt", "w")
println(A_io, ((N-1)^2)*((N-1)^2))
for i = 1:((N-1)^2)*((N-1)^2)
    println(A_io, A_Dense[i])
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
b_io = open("b_025.txt", "w")
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

ans_io = open("ans_05.txt", "w")
println(ans_io, (N-1)^2)
for i = 1:(N-1)^2
    println(ans_io, ANS[i])
end
close(ans_io)

#0 = Au + b
# direct solve
U = A\B
check = U-ANS
@show check_err = sqrt(check' * check) * sqrt(Δx*Δy)
# replace U with the cg solve
#U = cg(A,B; log=True)


open("xoutput.txt") do f
    n = readline(f)
    n = parse(Int64, n)
    global Output = zeros(n)
    global linect = 0

    while ! eof(f)
        l = readline(f)
        l = parse(Float64, l)
        global linect += 1
        Output[linect] = l
    end
end

# Check Convergence
println(size(Output))
#Output = reshape(Output, (N-1,N-1))
#println(size(Output))
diff = Output - ANS
@show err = sqrt(diff' * diff) * sqrt(Δx*Δy)


# plot error, plot exact vs. numeric
outexact = exact(x[2:N], y[2:N])
print(size(outexact))
@show plot(x[2:N], y[2:N], outexact, st=:surface)
# Numeric/Approximate
@show plot(x[2:N], y[2:N], Output, st=:surface)

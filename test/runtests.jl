# A package of useful testing commands
using Test

# We need to tell this test to load the module we are testing
using project1

# The Random package is useful for making tests as arbitrary as possible
using Random

# The LinearAlgebra package is useful for some norms
using LinearAlgebra

@testset "Cell centered data" begin

nx = 50; ny = 25
p = CellData(rand(nx+2,ny+2))

@test size(p) == (nx+2,ny+2)

@test typeof(p) <: ScalarData{nx,ny}

@test typeof(p) <: GridData


p = CellData(nx,ny)
# choose a random interior point to set equal to 1
i, j = rand(2:nx+1), rand(2:ny+1)
p[i,j] = 1.0
@test p[i,j] == p.data[i,j]

@test norm(p)*sqrt(nx*ny) == 1.0

# Test that subtraction of cell data works
p2 = deepcopy(p)
@test norm(p2 - p) == 0.0

end

@testset "Edge data" begin

nx = 50; ny = 25
p = CellData(rand(nx+2,ny+2))
q = EdgeData(p)

@test size(q.qx) == (nx+1,ny+2)
@test size(q.qy) == (nx+2,ny+1)

@test typeof(q) <: VectorData{nx,ny}
@test typeof(q) <: GridData


@test typeof(q.qx) <: ScalarData{nx,ny}
@test typeof(q.qy) <: ScalarData{nx,ny}

i, j = rand(2:nx+1), rand(2:ny+1)
v = rand()
q.qx[i,j] = v
@test q.qx.data[i,j] == v

# Test that subtraction of edge data works
q2 = deepcopy(q)
@test norm(q2 - q) == 0.0

# Test direct indexing
q = EdgeData(p)
q[100] = rand()
q[length(q.qx)+100] = rand()

@test q[100] == q.qx[100]
@test q[length(q.qx)+100] == q.qy[100]

end

@testset "Node data" begin

nx = 50; ny = 25
p = NodeData(nx,ny)

@test size(p) == (nx+1,ny+1)

@test typeof(p) <: ScalarData{nx,ny}

# Test that subtraction of node data works
p2 = deepcopy(p)
@test norm(p2 - p) == 0.0

end

@testset "1D Cell to Edge Differencing" begin

p = CellData(64,64)
q = EdgeData(p)
q.qx .= 2:66

diff!(p,q.qx)
@test norm(p) == 1.0

transpose(q.qy) .= 1:65

p = CellData(64,64)
diff!(p,q.qy)
@test norm(p) == 1.0

diff!(q.qx,p)
diff!(q.qy,p)


end

@testset "1D Node to Edge Differencing" begin

s = NodeData(64,64)
q = EdgeData(s)
q.qx .= 1:65
q.qy .= 1:66
diff!(s,q.qx)

@test norm(s) == 0.0

s = NodeData(64,64)
diff!(s,q.qy)
@test norm(s) == 1.0

diff!(q.qx,s)
diff!(q.qy,s)


end

@testset "Divergence" begin

nx = 50; ny = 25
q = EdgeData(nx,ny)
# choose a random edge in interior of grid and set it to 1
i, j = rand(2:nx), rand(2:ny+1)
q.qx[i,j] = 1

p = divergence(q)
# Result should be of type CellData
@test typeof(p) <: CellData

# Should not be able to accept other types of data
@test_throws MethodError divergence(p)

# Test that divergence is sum of 1-d differences
p2 = CellData(q)
diff!(p2,q.qx)
dqdy = CellData(q)
diff!(dqdy,q.qy)
p2 .+= dqdy
@test norm(p-p2) < 1e-13


end

@testset "Rot" begin

nx = 50; ny = 25
q = EdgeData(nx,ny)
fill!(q.qx,1.0)
w = rot(q)

@test typeof(w) <: NodeData

# The rot of constant data should be uniformly 0
@test norm(w) == 0.0

# Should not be able to accept other types of data
@test_throws MethodError rot(w)

# Test that rot is sum of 1-d differences
w2 = NodeData(q)
diff!(w2,q.qx)
w2 .= -w2
dqdx = NodeData(q)
diff!(dqdx,q.qy)
w2 .+= dqdx
@test norm(w-w2) < 1e-13

end

@testset "Gradient" begin

nx = 50; ny = 25
p = CellData(nx,ny)
fill!(p,1.0)
q = gradient(p)

@test typeof(q) <: EdgeData

# The gradient of constant data should be uniformly 0
@test norm(q) == 0.0

# Should not be able to accept other types of data
@test_throws MethodError gradient(q)

end

@testset "Curl" begin

nx = 50; ny = 25
s = NodeData(nx,ny)
fill!(s,1.0)
q = curl(s)

@test typeof(q) <: EdgeData

# The curl of constant data should be uniformly 0
@test norm(q) == 0.0

# Should not be able to accept other types of data
@test_throws MethodError curl(CellData(s))

end

@testset "Laplacian" begin

# test that Laplacian of cell data is equivalent to divergence of gradient
nx = 50; ny = 25
p = CellData(nx,ny)
p .= randn(size(p))
lapp = laplacian(p)
lapp2 = divergence(gradient(p))
err = CellData(lapp)
err .= lapp - lapp2
@test norm(err) < 1e-13

# test that second derivatives in each direction add up to full Laplacian

lapp2 = laplacian(p,1) + laplacian(p,2)
err .= lapp - lapp2
@test norm(err) < 1e-13


end

@testset "Nullspaces" begin

nx = 50; ny = 25

# set up some random node data
s = NodeData(nx,ny)
s .= randn(size(s))

# take the divergence of the curl
p = divergence(curl(s))

# Result should be very small
@test norm(p) < 1e-14

# do the same test with rot of the gradient
p = CellData(nx,ny)
p .= randn(size(p))

s = rot(gradient(p))
@test norm(s) < 1e-14

end

@testset "Interpolations" begin

# Interpolate edge data to edge data (x -> y, y -> x)
nx = 5; ny = 3
p = CellData(nx,ny)
p .= 1:7

q = EdgeData(p)
interpolate!(q,p)

@test q.qx[3,4] == 3.5
@test q.qy[3,4] == 3.0

q = EdgeData(64,64)
q.qx .= 1:65
q.qy .= 1:66

s = NodeData(q)
interpolate!(s,q.qx)
@test s[34,45] == 34


s = NodeData(q)
interpolate!(s,q.qy)
@test s[34,45] == 34.5



end

@testset "TimeMarching" begin

# Set up the state vector's initial condition
u = [1.0]

# set up right-hand side of du/dt = -t^2
function f(u,t)
  du = deepcopy(u)
  du[1] = -t^2
  return du
end

# exact solution
uex(t) = 1 - t^3/3

# set up the integrator
Δt = 0.01
rk = RK(u,Δt,f;rk=RK4)

t = 0.0
uarray = [u[1]]
tarray = [t]
for j = 1:100
    t, u = rk(t,u)
    push!(uarray,u[1])
    push!(tarray,t)
end

@test norm(uarray-uex.(tarray)) < 1e-14


end

@testset "Prolongation" begin

  NX = 4
  NY = 4
  uh = CellData(NX,NY)
  u2h = CellData(Int(NX/2)+1,Int(NY/2))
  @test_throws AssertionError prolong!(uh,u2h)

  # test that restriction and prolongation are transposes of one another
  # (even with boundary conditions set)
  NX = 16
  NY = 16
  uh = CellData(NX,NY)
  u2h = CellData(Int(NX/2),Int(NY/2))

  vh = CellData(uh)
  v2h = CellData(u2h)

  uh .= randn(size(uh))
  apply_bc!(uh)

  v2h .= randn(size(v2h))
  apply_bc!(v2h)

  prolong!(vh,v2h)
  restrict!(u2h,uh)
  @test abs(dot(uh,vh)-dot(u2h,v2h)) < 1e-14

end

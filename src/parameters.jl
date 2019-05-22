# set up parameter list types

export Parameters, VectorPoissonParameters, ScalarDirichletParameters, NSParameters,
         VectorDirichletParameters, ScalarNeumannParameters

abstract type Parameters end

"""
    ScalarDirichletParameters(Δx,x0,y0,uL,uR,uB,uT)

Set the problem parameters for a scalar-valued problem
with Dirichlet boundary conditions
"""
struct ScalarDirichletParameters <: Parameters

  # numerical parameters
  "grid spacing"
  Δx :: Real
  "lower left hand corner of domain"
  x0 :: Real
  y0 :: Real

  # boundary condition functions
  uL
  uR
  uB
  uT

end

"""
    ScalarNeumannParameters(Δx,x0,y0,qL,qR,qB,qT)

Set the problem parameters for a scalar-valued problem
with Neumann boundary conditions
"""
struct ScalarNeumannParameters <: Parameters

  # numerical parameters
  "grid spacing"
  Δx :: Real
  "lower left hand corner of domain"
  x0 :: Real
  y0 :: Real

  # boundary condition functions
  qL
  qR
  qB
  qT

end



"""
    VectorPoissonParameters(Δx,x0,y0,uL,uR,uB,uT,vL,vR,vB,vT)

Set the problem parameters for a scalar-valued problem
with Dirichlet boundary conditions
"""
struct VectorPoissonParameters <: Parameters

  # numerical parameters
  "grid spacing"
  Δx :: Real
  "lower left hand corner of domain"
  x0 :: Real
  y0 :: Real

  # boundary condition functions
  uL
  uR
  uB
  uT
  vL
  vR
  vB
  vT

end

"""
    NSParameters(ν,Δx,x0,y0,Δt,uL,uR,uB,uT,vL,vR,vB,vT)

Set the Navier-Stokes problem parameters
"""
struct NSParameters <: Parameters

  # physical parameters
  "viscosity"
  ν :: Real

  # numerical parameters
  "grid spacing"
  Δx :: Real
  "lower left hand corner of domain"
  x0 :: Real
  y0 :: Real
  "time step size"
  Δt :: Real
  "grid Fourier number"
  Fo :: Real

  # boundary condition functions
  uL
  uR
  uB
  uT
  vL
  vR
  vB
  vT

end

function NSParameters(ν,Δx,x0,y0,Δt,uL,uR,uB,uT,vL,vR,vB,vT)
    Fo = ν*Δt/Δx^2
    return NSParameters(ν,Δx,x0,y0,Δt,Fo,uL,uR,uB,uT,vL,vR,vB,vT)
end

VectorDirichletParameters = Union{VectorPoissonParameters,NSParameters}

#=
File: ps8-main.jl
Author: Aaron C Watt
Date: 2022-11-10
Purpose: Solve problems from 202AFall2022PS8.pdf

References:
https://alisdairmckay.com/Notes/NumericalCrashCourse/FuncApprox.html
https://alisdairmckay.com/Notes/NumericalCrashCourse/index.html

Expectations:
https://quantecon.github.io/Expectations.jl/dev/

Final version in ps8-notebook.jl
=#

#==============================================================================
                Packages
==============================================================================#
using Optim
using Parameters
using Plots
# using Expectations, Distributions
using Revise
includet("Tauchen.jl")


#==============================================================================
                Part A
===============================================================================
Write the household’s problem recursively.
Be sure to state what variables are chosen and all the constraints.
=#









#==============================================================================
                Part B
===============================================================================
Write a version of McKay’s PolyBasis function for this problem. 
Use a 2nd order polynomial basis (which will have 6 terms).
=#
PolyBasis(A, lnY) = [ones(size(A)) A lnY A.^2 A.*lnY lnY.^2]  # n x 6






#==============================================================================
                Part C
===============================================================================
Write a version of McKay’s PolyGetCoeff function for this problem.
=#
PolyGetCoef(A, lnY, V) = PolyBasis(A, lnY) \ V  # 6






#==============================================================================
                Part D
===============================================================================
Start your main Matlab program by reading in the parameter values into a structure.
=#
# Parameters Given
@consts begin
    γ = 2
    β = 0.94
    μ = 1
    bb = 0.4
    ρ = 0.9
    σ² = 0.01
    σ = σ²^(1/2)
    r = 0.05
    Z = 20
end
α = 1/3  # allow α to change for later problems



# Labor Income
# Y(Yₜ₋₁, εₜ; ρ=ρ, μ=μ) = exp( (1 − ρ)*log(μ) + ρ*log(Yₜ₋₁) + εₜ )
lnY(lnYₜ₋₁, εₜ; μ=μ) = (1 − ρ)*log(μ) + ρ*lnYₜ₋₁ + ε










#==============================================================================
                Part E
===============================================================================
Calculate the steady-state of A using the first-order conditions of the sequence
problem and the budget constraint. Then create an equally spaced grid from 0.05Ā 
to 1.95Ā, where Ā is the steady state.
Create a grid on A and log Y . Feel free to use McKay’s tauchen function as needed.
Use 7 grid points for log Y and 100 grid points for A. 
    A=K     Z=lnY
=#




# Create a grid for lnY
n_lnY = 7;  # number of points in our grid for lnY
n_sdlnY = 2;  # number of standard deviations to cover with the grid
# Note that we need to use log(μ) due to the formula used in tauchen()
GridlnY, GridPlnY = tauchen(n_lnY, log(μ), ρ, σ, n_sdlnY)
# tauchen() imported from Tauchen.jl

GridPlnY = GridPlnY'  # this is a 7 x 7 transition matrix for which the columns sum to 1
# the (i,j) element is the probability of moving from j to i.

# Using the BC and EE, solve for the steady state asset level
Aₛₛ = Z*(β * Z * α)^(α/(1-α))

# Create a grid from 0.05Aₛₛ to 1.95Aₛₛ
n_A = 100  # number of points in our grid for A
GridA = range(0.05*Aₛₛ, 1.95*Aₛₛ, n_A)

# Cartisian product of the grids, then decompose
AY = [(a,y) for a ∈ GridA for y ∈ GridlnY]
AA = [a for (a,y) ∈ AY]
lnYY = [y for (a,y) ∈ AY]





#==============================================================================
                Part F
===============================================================================
Write a version of McKay’s Bellman Matlab function for this problem. 
To do this, you will need to solve for consumption Ct as a function 
of the variables At , At+1 , Yt.
At+1 = Z(Yt − Ct + At)^α 
=#

# Utility function
U(C) = C^(1-γ)/(1-γ)

# Just this period's income
f(lnYₜ, Aₜ; α=α) = exp(lnYₜ) + Aₜ
# Budget Constraint defines Cₜ(Yₜ, Aₜ, Aₜ₊₁) = income - savings
# savings = Sₜ = (Aₜ₊₁/Z)^(1/α) from the savings technology
savings(Aₜ₊₁; α=α) = (Aₜ₊₁/Z)^(1/α)
c(lnYₜ, Aₜ, Aₜ₊₁; α=α) = f(lnYₜ, Aₜ; α=α) - savings(Aₜ₊₁; α=α)
# Maximum A' could be for C>0, given Y and A
max_Ap(lnYₜ, Aₜ; α=α) = Z*(exp(lnYₜ) + Aₜ)^α

"""
V = Bellman(b, Aₜ, lnY, Aₜ₊₁; α=α)
  Evaluate the RHS of the Bellman equation

    Inputs
    b     6 x 1 coefficients in polynomial for E[ V(A',lnY') | lnY ]
    Aₜ     n-vector of current assets A
    lnY   n-vector of current labor income
    Aₜ₊₁   n-vector of this period's savings (A')
    α     scalar savings technology parameter

    Output
    V     n-vector of value function
"""
function Bellman(b, Aₜ::AbstractArray, lnY::AbstractArray, Aₜ₊₁::AbstractArray; α=α)
    
    C = c.(lnY, Aₜ, Aₜ₊₁; α=α)
    u = U.(C)
    V = u .+ β * (PolyBasis(Aₜ, lnY) * b)
    return V
end
function Bellman(EV, Aₜ::Real, lnY::Real, Aₜ₊₁::Real; α=α)
    C = c(lnY, Aₜ, Aₜ₊₁; α=α)
    u = U(C)
    V = u + β * EV
    return V
end


#= testing the Bellman output
B(Y) = Bellman(zeros(6), [Aₛₛ], [Y], [Aₛₛ])[1]
p1_ = plot(GridlnY, B.(GridlnY))

B2(Ap) = Bellman(zeros(6), [Aₛₛ], [0], [Ap])[1]
p2_ = plot(GridA, B2.(GridA))

plot(p1_, p2_, layout=(2,1))


max_Ap(Aₛₛ, 70)
=#

#==============================================================================
                Part G
===============================================================================
Write a version of McKay’s MaxBellman function for this problem. Similarly to 
McKay’s program make sure to bound the upper constraint such that consumption
is positive. Additionally, make sure that the choices of next period’s assets
are bounded to be within the upper and lower bounds of the grid.
=#



"""Update elements of A with elments from B if the index I is 1 for that element"""
function update_A_with_B!(A, B, I)
    for i ∈ eachindex(I)
        if I[i]==1
            A[i] = B[i]
        end
    end
end




#*****************************************************************************
#*****************************************************************************
# The below functions do not seem to converge on the correct value function
# See the next section beginning with the MaxBellman function for the code
# that copies McKay. It seems to get something more reasonable, but still
# outputs that the policy function is constant at the lower bound.
# Probably an error in how I set the bounds of the maximization?
#*****************************************************************************
#*****************************************************************************



############### The next three functions are my own julia maximization
# The versions closer to McKay’s code are in the next section

""""Maximize the Bellman function using Aₜ₊₁ given b, Aₜ, lnY scalars"""
function MyMaxSingleBellman(b, A, lnY; lbA=first(AA), ubA=last(AA), α=α)
    to_minimize(Ap) = -Bellman(b, A, lnY, Ap; α=α)[1]  # only one value, need to extract
    # Want there to be >0 consumption, so put upper bound
    # at maximum A' that results in C>0, given lnY, Aₜ
    upperA = min(ubA, max_Ap(lnY, A; α=α) - 1e-3)
    # println("\nUB: $upperA,  LB: $lbA")
    out = optimize(to_minimize, lbA, upperA)
    minBell = out.minimum
    minA = out.minimizer
    return minA, minBell
end
function MyMaxSingleBellman2(EV, A, lnY; lbA=first(AA), ubA=last(AA), α=α)
    to_minimize(Ap) = -Bellman(EV, A, lnY, Ap; α=α)
    # Want there to be >0 consumption, so put upper bound
    # at maximum A' that results in C>0, given lnY, Aₜ
    upperA = min(ubA, max_Ap(lnY, A; α=α) - 1e-3)
    # println("\nUB: $upperA,  LB: $lbA")
    out = optimize(to_minimize, lbA, upperA)
    minBell = -1*out.minimum
    minA = out.minimizer
    return minA, minBell
end

""""Maximize the Bellman function using Aₜ₊₁ given b, Aₜ, lnY vectors"""
function MyMaxBellman(b; α=α)
    # Define the function taking scalar Aₜ, lnY
    MaxBellmanVector(Aₜ, lnY) = MyMaxSingleBellman(b, Aₜ, lnY; α=α)
    # Broadcast over this function
    out = MaxBellmanVector.(AA, lnYY)
    minA = [x[1] for x in out]
    minBell = [x[2] for x in out]
    println("var(minA) = $(var(minA)),  var(minBell) = $(var(minBell))")
    return minBell, minA
end
function MyMaxBellman2(EV; α=α)
    # Define the function taking scalar Aₜ, lnY
    # MaxBellmanVector(Aₜ, lnY) = MyMaxSingleBellman2(EV, Aₜ, lnY; α=α)
    # Broadcast over this function
    # out = MaxBellmanVector.(AA, lnYY)
    out = MyMaxSingleBellman2.(EV, AA, lnYY)
    minA = [x[1] for x in out]
    minBell = [x[2] for x in out]
    println("var(minA) = $(var(minA)),  var(minBell) = $(var(minBell))")
    return minBell, minA
end

"""Iterate over the polynomial coefficients to converge on the value function"""
function MyBellmanIteration()
    # initial guess of the coefficients of the polynomial approx to the value function (zero function)
    b = zeros(6)
    Aₜ₊₁0 = zeros(size(AA));
    MAXIT = 2000;
    Vlist, Alist, blist = [zeros(size(AA))], [zeros(size(AA))], [b]
    for it = 1:MAXIT

        V, Aₜ₊₁ = MyMaxBellman(b; α=α)
        append!(Vlist, [V])
        append!(Alist, [Aₜ₊₁])

        # take the expectation of the value function from the perspective of the previous Z
        # Need to reshape V into a 20 by 7 array where the rows correspond different levels
        # of assets and the columns correspond to different levels of income.
        # need to take the dot product of each row of the array with the appropriate column of the Markov chain transition matrix
        EV = reshape(V, n_A, n_lnY) * GridPlnY

        # update our polynomial coefficients
        b = PolyGetCoef(AA, lnYY, EV[:])
        append!(blist, [b])

        # see how much our policy rule has changed
        Atest = maximum(abs.(Aₜ₊₁0 .- Aₜ₊₁))
        Vtest = maximum(abs.(Vlist[end] - Vlist[end-1]))
        btest = maximum(abs.(blist[end] - blist[end-1]))

        Aₜ₊₁0 = copy(Aₜ₊₁)
        println("mean(A) = $(mean(Aₜ₊₁)),   Var(A) = $(var(Aₜ₊₁))")

        println("iteration $it, Atest = $Atest, Vtest = $Vtest, btest = $btest")
        if Vtest < 1e-5
            break
        end
    end
    return Alist, Vlist, blist
end
function MyBellmanIteration2()
    # initial guess of the coefficients of the polynomial approx to the value function (zero function)
    b = zeros(6)
    EV = Aₜ₊₁0 = zeros(size(AA));
    MAXIT = 2000;
    Vlist, Alist, blist = [zeros(size(AA))], [zeros(size(AA))], [b]
    for it = 1:MAXIT

        println(size(EV), size(AA), size(lnYY))
        V, Aₜ₊₁ = MyMaxBellman2(EV; α=α)
        append!(Vlist, [V])
        append!(Alist, [Aₜ₊₁])

        # take the expectation of the value function from the perspective of the previous lnY
        # Need to reshape V into a 20 by 7 array where the rows correspond different levels
        # of assets and the columns correspond to different levels of income.
        # need to take the dot product of each row of the array with the appropriate column of the Markov chain transition matrix
        # So we are taking the expectation of the value function by multiplying each lnY level's value of the 
        # value function by the probability that we will still be in that level next period.
        EV = reshape(V, n_A, n_lnY) * GridPlnY
        EV = EV[:]

        # update our polynomial coefficients
        b = PolyGetCoef(AA, lnYY, EV)
        append!(blist, [b])

        # see how much our policy rule has changed
        Atest = maximum(abs.(Aₜ₊₁0 .- Aₜ₊₁))
        Vtest = maximum(abs.(Vlist[end] - Vlist[end-1]))
        btest = maximum(abs.(blist[end] - blist[end-1]))

        Aₜ₊₁0 = copy(Aₜ₊₁)
        println("mean(A) = $(mean(Aₜ₊₁)),   Var(A) = $(var(Aₜ₊₁))")

        println("iteration $it, Atest = $Atest, Vtest = $Vtest, btest = $btest")
        if Vtest < 1e-5
            break
        end
    end
    return Alist, Vlist, blist
end


#=
# TESTING THE MyBellmanIteration Output
@time Alist, Vlist, blist = MyBellmanIteration2()

# @time Alist, Vlist, blist = MyBellmanIteration()
Aₜ₊₁_, V_, b_ = last(Alist), last(Vlist), last(blist)
ii = length(Vlist)
plot(AA, Aₜ₊₁_)
# Value Function
surface(AA, lnYY, Vlist[1], camera=(-10,30), xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, lnYY, Vlist[2], camera=(-10,30), xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, lnYY, Vlist[3], camera=(-10,30), xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, lnYY, Vlist[10], camera=(-10,30), xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, lnYY, Vlist[50], camera=(-10,30), xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, lnYY, Vlist[100], camera=(-10,30), xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, lnYY, Vlist[130], camera=(-10,30), xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, lnYY, V_, camera=(10,50), xlabel="A", ylabel="lnY", zlabel="V")



# Policy Function
surface(AA, lnYY, Aₜ₊₁_, camera=(-10,80), xlabel="A", ylabel="lnY", zlabel="A'")
# Consumption
CC = c.(lnYY, AA, Aₜ₊₁_)
surface(AA, lnYY, CC, camera=(-10,30), xlabel="A", ylabel="lnY", zlabel="C")
surface(AA, lnYY, CC, camera=(-10,30), xlims=(26,28.5), xlabel="A", ylabel="lnY", zlabel="C")

=#


#= Notes and code checks
a = MyMaxBellman(b)

ii = 1
MyMaxSingleBellman(b, GridA[ii], GridlnY[ii]; α=α)
MyMaxSingleBellman(b, GridA[ii], GridlnY[ii], first(GridA); α=α)
Bellman(b, GridA[ii], GridlnY[ii], 29.27243188437049; α=α)
Bellman(b, GridA[ii], GridlnY[ii], -37688.493909217614; α=α)

Bellman(b, GridA[ii], GridlnY[ii], first(GridA); α=α)
Bellman(b, GridA[ii], GridlnY[ii], [first(GridA)][1]; α=α)

g(x) = Bellman(b, GridA[ii], GridlnY[ii], x[1]; α=α)[1]
x0 = [first(GridA)]
g(x0)
optimize(g, x0)

# recall how to maximize a function using Optim.jl
g(x) = (x[1]-4)^2  # Concave down function, has a max
x0 = [0.0]
out = optimize(g, x0)

g(x) = (x-4)^2  # Concave up function, has a min
# Define lower and upper bounds to the search
lb, ub = -5, 20
# run the optimization routine
out = optimize(g, lb, ub)
out.minimizer  # minimizing x solution
out.minimum  # minimized function value
=#





############### Direct translations of McKay's code

"""
[V, Ap] = MaxBellman(b,Grid)
    Maximizes the RHS of the Bellman equation using golden section search

    Inputs
    b         6-vector coefficients in polynomial for E[ V(K',Z') | Z ]

    Globals
    GridlnY     initial vector of lnY
    GridPlnY    transition matrix of lnY
    GridA       initial vector of A
    AY          cartisian product of AxlnY
    AA          decomposed A vector from AY
    lnYY        decomposed lnY vector from AY
"""
function MaxBellman(b; α=α)

    p = (sqrt(5)-1)/2
    
    A = first(GridA) .* ones(size(AA))
    # f(lnYₜ, Aₜ; α=α) -> f.(lnYY, AA; α=α)
    D = min.(max_Ap.(lnYY, AA; α=α) .- 1e-3, last(GridA)) # -1e-3 so we always have positve consumption.
    B = p*A .+ (1-p)*D
    C = (1-p)*A .+ p*D
    
    fB = Bellman(b, AA, lnYY, B)
    fC = Bellman(b, AA, lnYY, C)
    
    MAXIT = 1000;
    for it_inner = 1:MAXIT
    
        if all(D-A .< 1e-6)
            break
        end
    
        I = fB .> fC
        
        # D[I] = C[I]
        update_A_with_B!(D, C, I)
        # C[I] = B[I];
        update_A_with_B!(C, B, I)
        # fC[I] = fB[I];
        update_A_with_B!(fC, fB, I)
        # B[I] = p*C[I] + (1-p)*A[I];
        update_A_with_B!(B, p*C .+ (1-p)*A, I)
        # fB[I] = Bellman(Par,b,Grid.KK(I),Grid.ZZ(I),B(I));
        update_A_with_B!(fB, Bellman(b, AA, lnYY, B), I)
    
        # A(~I) = B(~I);
        update_A_with_B!(A, B, .~I)
        # B(~I) = C(~I);
        update_A_with_B!(B, C, .~I)
        # fB(~I) = fC(~I);
        update_A_with_B!(fB, fC, .~I)
        # C(~I) = p*B(~I) + (1-p)*D(~I);
        update_A_with_B!(C, p*B .+ (1-p)*D, .~I)
        # fC(~I) = Bellman(Par,b,Grid.KK(~I),Grid.ZZ(~I),C(~I));
        update_A_with_B!(fC, Bellman(b, AA, lnYY, B), .~I)
    end
    
    # At this stage, A, B, C, and D are all within a small epsilon of one
    # another.  We will use the average of B and C as the optimal level of
    # savings.
    Aₜ₊₁ = (B .+ C) ./ 2;

    # Make sure that the choices of next period’s assets are bounded to 
    # be within the upper and lower bounds of the grid.
    Aₜ₊₁ = max.(Aₜ₊₁, first(GridA))  # lower bound
    Aₜ₊₁ = min.(Aₜ₊₁, last(GridA))   # upper bound
    
    # evaluate the Bellman equation at the optimal policy to find the new
    # value function.
    V = Bellman(b, AA, lnYY, Aₜ₊₁);
    return V, Aₜ₊₁
end

V__ = MaxBellman(zeros(6))[2]
A__ = MaxBellman(zeros(6))[1]















#==============================================================================
                Part H
===============================================================================
Write the value function iteration for-loop for this problem. Once the iterations
converge, plot the value function, policy function, and the consumption on the
meshgrid of (A, Y ) using the surf function in Matlab.
=#





function value_function_iteration()
    # initial guess of the coefficients of the polynomial approx to the value function (zero function)
    b = b0 = zeros(6,1)

    Aₜ₊₁0 = zeros(size(AA)); V0 = zeros(size(AA))
    V, Aₜ₊₁ = V0, Aₜ₊₁0
    MAXIT = 2000;
    ifinal = 0
    for it = 1:MAXIT

        V, Aₜ₊₁ = MaxBellman(b; α=α)

        # take the expectation of the value function from the perspective of the previous Z
        # Need to reshape V into a 20 by 7 array where the rows correspond different levels
        # of assets and the columns correspond to different levels of income.
        # need to take the dot product of each row of the array with the appropriate column of the Markov chain transition matrix
        EV = reshape(V, n_A, n_lnY) * GridPlnY

        # update our polynomial coefficients
        b = PolyGetCoef(AA, lnYY, EV[:])

        # see how much our policy rule has changed
        # test = maximum(abs.(Aₜ₊₁0 .- Aₜ₊₁))
        test = maximum(abs.(V0 .- V))
        # test = maximum(abs.(b0 .- b))
        b0 = b
        V0 = V
        Aₜ₊₁0 = Aₜ₊₁
        ifinal = it
        println("mean(A) = $(mean(Aₜ₊₁)),   Var(A) = $(var(Aₜ₊₁))")

        println("iteration $it, test = $test")
        if test < 1e-5
            break
        end
    end
    
    return Dict(:Ap => Aₜ₊₁, :V => V, :b => b, :i => ifinal)
end

out = value_function_iteration()
Ap = out[:Ap]; V = out[:V];

pyplot()
plot(AA, Ap)
surface(AA, lnYY, Ap, camera=(-160,30), xlabel="A", ylabel="lnY", zlabel="A'")
surface(AA, lnYY, V, camera=(-160,30), xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, lnYY, c.(lnYY, AA, Ap), camera=(-160,30), xlabel="A", ylabel="lnY", zlabel="C")
out[:b]

[V Ap]












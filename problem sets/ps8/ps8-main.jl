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
import Pkg
Pkg.activate(pwd())
try
    using Optim, Parameters, Plots, Revise, DataFrames, Pluto
    using Optim: maximum, maximizer
    pyplot()
catch e
    Pkg.add(["Plots", "PyPlot", "Optim", "Parameters", "Revise", "DataFrames", "Pluto"])
    using Optim, Parameters, Plots, Revise, DataFrames
    using Optim: maximum, maximizer
    pyplot()
end
includet("Tauchen.jl")  # Installs Distributions if not installed



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
PolyBasis(A::AbstractArray, lnY::AbstractArray) = [ones(size(A)) A lnY A .^ 2 A .* lnY lnY .^ 2]  # n x 6
PolyBasis(A::Real, lnY::Real) = [1 A lnY A^2 A * lnY lnY^2]  # 1 x 6






#==============================================================================
                Part C
===============================================================================
Write a version of McKay’s PolyGetCoeff function for this problem.
=#
PolyGetCoef(A, lnY, V) = PolyBasis(A, lnY) \ V  # 6 x 1






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
    σ = σ²^(1 / 2)
    r = 0.05
    Z = 20
end
α = 1 / 3  # allow α to change for later problems



# Labor Income
# Y(Yₜ₋₁, εₜ; ρ=ρ, μ=μ) = exp( (1 − ρ)*log(μ) + ρ*log(Yₜ₋₁) + εₜ )
lnY(lnYₜ₋₁, εₜ) = (1 - ρ) * log(μ) + ρ * lnYₜ₋₁ + εₜ










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
nY = 7;  # number of points in our grid for lnY
n_sdlnY = 2;  # number of standard deviations to cover with the grid
# Note that we need to use log(μ) due to the formula used in tauchen()
GridlnY, GridPlnY = tauchen(nY, log(μ), ρ, σ, n_sdlnY)
# tauchen() imported from Tauchen.jl

GridPlnY = GridPlnY'  # this is a 7 x 7 transition matrix for which the columns sum to 1
# the (i,j) element is the probability of moving from j to i.

# Using the BC and EE, solve for the steady state asset level
Aₛₛ = Z * (β * Z * α)^(α / (1 - α))

# Create a grid from 0.05Aₛₛ to 1.95Aₛₛ
nA = 100  # number of points in our grid for A
GridA = range(0.05 * Aₛₛ, 1.95 * Aₛₛ, length=nA)

# Cartisian product of the grids, then decompose
AY = [(a, y) for y ∈ GridlnY for a ∈ GridA]
AA = [a for (a, y) ∈ AY]
YY = [y for (a, y) ∈ AY]




#==============================================================================
                Part F
===============================================================================
Write a version of McKay’s Bellman Matlab function for this problem. 
To do this, you will need to solve for consumption Ct as a function 
of the variables At , At+1 , Yt.
At+1 = Z(Yt − Ct + At)^α 
=#

# Utility function
U(C) = C^(1 - γ) / (1 - γ)

# Just this period's income
f(lnYₜ, Aₜ; α=α) = exp(lnYₜ) + Aₜ
# Budget Constraint defines Cₜ(Yₜ, Aₜ, Aₜ₊₁) = income - savings
# savings = Sₜ = (Aₜ₊₁/Z)^(1/α) from the savings technology
savings(Aₜ₊₁; α=α) = (Aₜ₊₁ / Z)^(1 / α)
c(lnYₜ, Aₜ, Aₜ₊₁; α=α) = f(lnYₜ, Aₜ; α=α) - savings(Aₜ₊₁; α=α)
# Maximum A' could be for C>0, given Y and A
Aprime(lnYₜ, Aₜ, Cₜ; α=α) = Z * (exp(lnYₜ) + Aₜ - Cₜ)^α
max_Ap(lnYₜ, Aₜ; α=α) = Aprime(lnYₜ, Aₜ, 0; α=α)

"""
V = Bellman(b, Aₜ, lnYₜ, Aₜ₊₁; α=α)
  Evaluate the RHS of the Bellman equation

    Inputs
    b     6 x 1 coefficients in polynomial for E[ V(A',lnY') | lnY ]
    Aₜ     n-vector of current assets A
    lnYₜ   n-vector of current labor income
    Aₜ₊₁   n-vector of this period's savings (A')
    α     scalar savings technology parameter

    Output
    V     n-vector of value function
"""
function Bellman(b::AbstractArray, Aₜ::Real, lnYₜ::Real, Aₜ₊₁::Real; α=α)
    # Scalar A' and lnY, vector of coefficients b
    C = c(lnYₜ, Aₜ, Aₜ₊₁; α=α)
    u = U(C)
    V = u + β * (PolyBasis(Aₜ₊₁, lnYₜ)*b)[1]
    return V
end
function Bellman(b::AbstractArray, Aₜ::AbstractArray, lnYₜ::AbstractArray, Aₜ₊₁::AbstractArray; α=α)
    # Vector A' and lnY, vector of coefficients b
    C = c.(lnYₜ, Aₜ, Aₜ₊₁; α=α)
    u = U.(C)
    V = u .+ β * (PolyBasis(Aₜ₊₁, lnYₜ) * b)
    return V
end
function Bellman(EV::Real, Aₜ::Real, lnYₜ::Real, Aₜ₊₁::Real; α=α)
    # Scalar A' and lnY, scalar EV
    C = c(lnYₜ, Aₜ, Aₜ₊₁; α=α)
    u = U(C)
    V = u + β * EV
    return V
end


#= testing the Bellman output
B(Y) = Bellman(zeros(6), Aₛₛ, Y, Aₛₛ)
p1_ = plot(GridlnY, B.(GridlnY), xlabel="lnY", ylabel="Bellman")

B2(Ap) = Bellman(zeros(6), Aₛₛ, 0, Ap)
p2_ = plot(GridA, B2.(GridA), xlabel="A'", ylabel="Bellman", xlims=(0,72), ylims=(-0.1,0))
maxAp1 = max_Ap(0, Aₛₛ)
vline!(p2_, [maxAp1], label="max A'")
println("max A' given Aₛₛ and lnY=0: $(max_Ap(0, Aₛₛ))")

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
        if I[i] == 1
            A[i] = B[i]
        end
    end
end




#*****************************************************************************
#*****************************************************************************
# The below functions converge to the same value and policy functions as the
# the translation of McKay's code, but the following functions use Optim.jl's
# optimizer to find the maximizing A' instead of the golden search method.
#*****************************************************************************
#*****************************************************************************



############### The next three functions are my own julia maximization
# The versions closer to McKay’s code are in the next section

"""Maximize the Bellman function using Aₜ₊₁ given b, Aₜ, lnY scalars"""
function MyMaxSingleBellman(b, Aₜ, lnYₜ; lbA=first(AA), ubA=last(AA), α=α)
    to_maximize(Ap) = Bellman(b, Aₜ, lnYₜ, Ap; α=α)
    # Want there to be >0 consumption, so put upper bound
    # at maximum A' that results in C>0, given lnYₜ, Aₜ
    upperA = min(ubA, max_Ap(lnYₜ, Aₜ; α=α) - 1e-3)
    # println("\nUB: $upperA,  LB: $lbA")
    out = maximize(to_maximize, lbA, upperA)
    maxBell = maximum(out)
    maxA = maximizer(out)
    return maxA, maxBell
end

"""Maximize the Bellman function using Aₜ₊₁ given b, Aₜ, lnY vectors"""
function MyMaxBellman(b; α=α)
    # Define the function taking scalar Aₜ, lnY
    MaxBellmanVector(Aₜ, lnYₜ) = MyMaxSingleBellman(b, Aₜ, lnYₜ; α=α)
    # Broadcast over this function
    out = MaxBellmanVector.(AA, YY)
    maxA = [x[1] for x in out]
    maxBell = [x[2] for x in out]
    # println("var(maxA) = $(var(maxA)),  var(maxBell) = $(var(maxBell))")
    return maxBell, maxA
end

"""Iterate over the polynomial coefficients to converge on the value function"""
function MyBellmanIteration()
    # initial guess of the coefficients of the polynomial approx to the value function (zero function)
    b = zeros(6)
    Aₜ₊₁0 = zeros(size(AA))
    MAXIT = 2000
    Vlist, Alist, blist = [zeros(size(AA))], [zeros(size(AA))], [b]
    for it = 1:MAXIT
        # println("b = $b")
        V, Aₜ₊₁ = MyMaxBellman(b; α=α)
        append!(Vlist, [V])
        append!(Alist, [Aₜ₊₁])

        # take the expectation of the value function from the perspective of the previous A
        # Need to reshape V into a 100x7 array where the rows correspond different levels
        # of assets and the columns correspond to different levels of income.
        # need to take the dot product of each row of the array with the appropriate column of the Markov chain transition matrix
        EV = reshape(V, nA, nY) * GridPlnY

        # update our polynomial coefficients
        b = PolyGetCoef(AA, YY, EV[:])
        append!(blist, [b])

        # see how much our policy rule has changed
        Atest = maximum(abs.(Aₜ₊₁0 .- Aₜ₊₁))
        Vtest = maximum(abs.(Vlist[end] - Vlist[end-1]))
        btest = maximum(abs.(blist[end] - blist[end-1]))
        if max(Atest, Vtest, btest) < 1e-5
            break
        end
        
        Aₜ₊₁0 = copy(Aₜ₊₁)
    end
    return Alist, Vlist, blist
end


@time Alist, Vlist, blist = MyBellmanIteration();
println("MyBellmanIteration Finished in $(length(blist)-1) iterations");
Aₜ₊₁_, V_, b_ = last(Alist), last(Vlist), last(blist)
plotargs = (camera=(-45, 20), xlabel="lnY", ylabel="A",
            legend=:none, aspect_ratio=[1,1,2])
# Value Function
pH1 = surface(YY, AA, V_, title="Value Function"; plotargs...)
# Policy Function
pH2 = surface(YY, AA, Aₜ₊₁_, title="A' Policy Function"; plotargs...)
# Consumption
CC = c.(YY, AA, Aₜ₊₁_)
pH3 = surface(YY, AA, CC, title="Consumption"; plotargs...)

pH4 = plot(pH1, pH2, pH3, layout=(1,3), size=(1600, 800),
            tickfontsize=14, labelfontsize=16,
            xtickfontrotation = -30)
# Save figs
savefig(pH1, "H-valuefunction")
savefig(pH2, "H-policyfunction")
savefig(pH3, "H-consumption")
savefig(pH4, "H-all")












############### Direct translations of McKay's code
rnd(x) = round(x, digits=4)

"""
[V, Ap] = MaxBellman(b; α=α)
    Maximizes the RHS of the Bellman equation using golden section search

    Inputs
    b         6-vector coefficients in polynomial for E[ V(K',Z') | Z ]

    Globals
    GridlnY     initial vector of lnY
    GridPlnY    transition matrix of lnY
    GridA       initial vector of A
    AY          cartisian product of AxlnY
    AA          decomposed A vector from AY
    YY        decomposed lnY vector from AY
"""
function MaxBellman(b; α=α)
    p = (sqrt(5) - 1) / 2

    A = first(GridA) .* ones(size(AA))
    # f(lnYₜ, Aₜ; α=α) -> f.(lnYY, AA; α=α)
    D = min.(max_Ap.(YY, AA; α=α) .- 1e-3, last(GridA))
    # -1e-3 so we always have positve consumption.
    B = p * A .+ (1 - p) * D
    C = (1 - p) * A .+ p * D

    fB = Bellman(b, AA, YY, B)
    fC = Bellman(b, AA, YY, C)

    MAXIT = 1000
    for it_inner = 1:MAXIT
        # Stop loop if we have converged
        if all(D - A .< 1e-6)
            break
        end

        I = fB .> fC

        # for indicies where fB > fC
        update_A_with_B!(D, C, I)
        update_A_with_B!(C, B, I)
        update_A_with_B!(fC, fB, I)
        update_A_with_B!(B, p * C .+ (1 - p) * A, I)
        update_A_with_B!(fB, Bellman(b, AA, YY, B), I)

        # for indicies where fB <= fC
        update_A_with_B!(A, B, .~I)
        update_A_with_B!(B, C, .~I)
        update_A_with_B!(fB, fC, .~I)
        update_A_with_B!(C, p * B .+ (1 - p) * D, .~I)
        update_A_with_B!(fC, Bellman(b, AA, YY, C), .~I)
    end

    # At this stage, A, B, C, and D are all within a small epsilon of one
    # another.  We will use the average of B and C as the optimal level of
    # savings.
    Aₜ₊₁ = (B .+ C) ./ 2

    # Make sure that the choices of next period’s assets are bounded to 
    # be within the upper and lower bounds of the grid.
    Aₜ₊₁ = max.(Aₜ₊₁, first(GridA))  # lower bound
    Aₜ₊₁ = min.(Aₜ₊₁, last(GridA))   # upper bound

    # evaluate the Bellman equation at the optimal savings policy to find the new
    # value function.
    V = Bellman(b, AA, YY, Aₜ₊₁)
    return V, Aₜ₊₁
end

V__, A__ = MaxBellman(zeros(6))















#==============================================================================
                Part H
===============================================================================
Write the value function iteration for-loop for this problem. Once the iterations
converge, plot the value function, policy function, and the consumption on the
meshgrid of (A, Y ) using the surf function in Matlab.
=#


function value_function_iteration()
    # initial guess of the coefficients of the polynomial approx to the value function (zero function)
    b = b0 = zeros(6, 1)

    Aₜ₊₁0 = zeros(size(AA))
    V0 = zeros(size(AA))
    V, Aₜ₊₁ = V0, Aₜ₊₁0
    MAXIT = 2000
    ifinal = 0
    for it = 1:MAXIT

        V, Aₜ₊₁ = MaxBellman(b; α=α)

        # take the expectation of the value function from the perspective of the previous Z
        # Need to reshape V into a 20 by 7 array where the rows correspond different levels
        # of assets and the columns correspond to different levels of income.
        # need to take the dot product of each row of the array with the appropriate column of the Markov chain transition matrix
        EV = reshape(V, nA, nY) * GridPlnY

        # update our polynomial coefficients
        b = PolyGetCoef(AA, YY, EV[:])

        # see how much our policy rule has changed
        Atest = maximum(abs.(Aₜ₊₁0 .- Aₜ₊₁))
        Vtest = maximum(abs.(V0 .- V))
        btest = maximum(abs.(b0 .- b))
        b0 = b
        V0 = V
        Aₜ₊₁0 = Aₜ₊₁
        ifinal = it
        # println("mean(A) = $(mean(Aₜ₊₁)),   Var(A) = $(var(Aₜ₊₁))")

        # println("iteration $it, Vtest = $Vtest, Atest = $Atest, btest = $btest")
        if max(Vtest, Atest, btest) < 1e-5
            break
        end
    end

    return Dict(:Ap => Aₜ₊₁, :V => V, :b => b, :i => ifinal)
end

@time out = value_function_iteration();
println("value_function_iteration Finished in $(out[:i]) iterations");
Ap2 = out[:Ap];
V2 = out[:V];


# plot(AA, Ap2)
surface(AA, YY, Ap2, camera=angle1, xlabel="A", ylabel="lnY", zlabel="A'")
surface(AA, YY, V2, camera=angle1, xlabel="A", ylabel="lnY", zlabel="V")
surface(AA, YY, c.(YY, AA, Ap2), camera=angle1, xlabel="A", ylabel="lnY", zlabel="C")
out[:b]

[V2 Ap2]





#= PLOT THE RESULTS -- not need to turn in
DK = Grid.K/Kstar-1; % Capital grid as percent deviation from steady state

DKp = reshape(Kp,Grid.nK,Grid.nZ)./reshape(Grid.KK,Grid.nK,Grid.nZ) - 1;
  % savings policy rule as a 20 x 7 array expressed as a percent change from current K

plot(DK, DKp);  % plot the policy rule

hold on;        % next plots go on the same figure

plot(DK, zeros(Grid.nK,1), 'k--'); % add a zero line, k-- means black and dashsed

xlabel('K in % deviation from steady state')  % label the axes
ylabel('(K'' - K)/K')
=#

# Beginning-period assets as % deviation from steady state assets
DA2 = GridA/Aₛₛ .- 1

# savings policy rule as a 100 x 7 array expressed as a % change from Beginning-period assets
DAp2 = reshape(Ap2, nA, nY) ./ reshape(AA, nA, nY) .- 1

# Plot the policy rule
pH1 = plot(DA2, DAp2,
           xlabel="A in % deviation from steady state",
           ylabel="(A' - A)/A")
hline!(pH1, [0], label="", color="black")



#= INTERPOLATE OPTIMAL ASSET POLICY FOR POINTS NOT ON GRID
# McKay's code
bKp =  PolyGetCoef(Grid.KK,Grid.ZZ,Kp);
Kp2903 = PolyBasis(29,0.03) * bKp
C2903 = f(Par,29,0.03) - Kp2903

# Translation for point A=10, lnY=0.1
bAp2 = PolyGetCoef(AA, YY, Ap2)
Ap1001 = PolyBasis(10, 0.1) * bAp2
C1001 = c(0.1, 10, (PolyBasis(10, 0.1) * bAp2)[1])
=#













#==============================================================================
                Part I
===============================================================================
(Optional) Implement the Howard acceleration for this problem. Report the speed improvement that
you are able to achieve. (You will need to experiment with the number of iterations to figure out
what works well in terms of giving a speed improvement.)
=#


"""Iterate faster over the polynomial coefficients to converge on the value function using Howard acceleration."""
function MyFasterBellmanIteration(; inner_mod = 500)
    # initial guess of the coefficients of the polynomial approx to the value function (zero function)
    b = zeros(6)
    MAXIT = 20000; inner_it = 0
    Vlist, Alist, blist = [zeros(size(AA))], [zeros(size(AA))], [b]
    for it = 1:MAXIT
        # Every inner_mod iterations, get the maximizing A'
        if it % inner_mod == 1
            V, Aₜ₊₁ = MyMaxBellman(b; α=α)
            append!(Vlist, [V])
            append!(Alist, [Aₜ₊₁])
            inner_it += 1
        else
            V = Bellman(b, AA, YY, Alist[end])
        end

        # take the expectation of the value function from the perspective of the previous A
        # Need to reshape V into a 100x7 array where the rows correspond different levels
        # of assets and the columns correspond to different levels of income.
        # need to take the dot product of each row of the array with the appropriate column of the Markov chain transition matrix
        EV = reshape(V, nA, nY) * GridPlnY

        # update our polynomial coefficients
        b = PolyGetCoef(AA, YY, EV[:])
        append!(blist, [b])

        # see how much our policy rule has changed
        Atest = maximum(abs.(Alist[end] - Alist[end-1]))
        Vtest = maximum(abs.(Vlist[end] - Vlist[end-1]))
        btest = maximum(abs.(blist[end] - blist[end-1]))
        if max(Atest, Vtest, btest) < 1e-5
            println("Converged in $it iterations, $inner_it maximization iterations")
            break
        end        
    end
    return Alist, Vlist, blist
end

# warm up function (precompile)
MyFasterBellmanIteration(inner_mod=100);

# Find which modulo for the bellman A' maximization results in shortest time
mods = 10:10:1000
f(x) = @elapsed MyFasterBellmanIteration(inner_mod=x);
times = f.(mods)
mintime0, minidx = findmin(times)
minmod = mods[minidx]

# Compare to 
mintime1 = @elapsed MyBellmanIteration()
multiplier = round(mintime1 / mintime0, digits=2)
println("Howard acceleration with mod $minmod resulted in $multiplier times faster convergence")


















#==============================================================================
                Part J
===============================================================================
Adapt McKay’s Simulate function for this problem.
=#

struct SimReturn
    Ap
    A
    Y
    C
end

"""
Sim = Simulate(bKp, Mode, T)
    Simulates the model.
    Inputs:
    bKp       Polynomial coefficients for polynomial for Kp policy rule
    Mode      Mode = 'random' -> draw shocks
              Mode = 'irf'    -> impulse response function
    T         # of periods to simulate
"""
function Simulate(bAp, Mode, T; α=α)
    A = zeros(T); Y = zeros(T)
    A[1] = Aₛₛ
    
    if Mode == "irf"
        Y[1] = σ
        ε = zeros(T)
    elseif Mode == "random"
        Y[1] = 0
        ε = σ * randn(T)
    else
        throw("Unrecognized Mode $Mode in Simulate()");
    end
    
    Ap1 = PolyBasis(A[1:(T-1)], Y[1:(T-1)]) * bAp
    # display(sum(Ap1 .> max_Ap.(Y[1:(T-1)], A[1:(T-1)])))
    # A[2:T] = min.(Ap1, max_Ap.(Y[1:(T-1)], A[1:(T-1)]))


    for t ∈ 2:T
        Aptemp = (PolyBasis(A[t-1], Y[t-1]) * bAp)[1]
        Apmax = max_Ap(Y[t-1], A[t-1])
        # A[t] = min(Aptemp, Apmax)
        # println(Apmax<Aptemp)
        A[t] = (PolyBasis(A[t-1], Y[t-1]) * bAp)[1]
        Y[t] = lnY(Y[t-1], ε[t]) 
    end
    
    # Compute quantities from state variables
    Ti = 2:(T-1)
    Ap2 = A[Ti .+ 1]
    A = A[Ti]
    Y = Y[Ti]
    # Y = f(Par,A,Y) - (1-δ) * A;
    C = c.(Y, A, Ap2; α=α)

    return SimReturn(Ap2, A, Y, C)
    
end














#==============================================================================
                Part K
===============================================================================
Using your new Simulate function, produce a 10,000 period simulation of the evolution of A, Y ,
and C. Report a histogram of A, Y , and C. (Note: Yt not log Yt .) Report the mean and standard
deviation of each variable. Plot the evolution of A, Y , and C over a 100 period stretch starting from
period 1000. How do these mean values compare to the steady-state values calculated earlier?
=#

bAp2 = PolyGetCoef(AA, YY, Ap2)
nPeriods = 10000
@time SimK = Simulate(bAp2, "random", nPeriods);

# Report a histogram of A, Y , and C. (Note: Yt not log Yt .) 
hk1 = histogram(SimK.A, title="A", label="", xlims=(61,64))
hk2 = histogram(ℯ.^SimK.Y, title="Y", label="")
hk3 = histogram(SimK.C, title="C", label="", xlims=(31,35))
hk4 = plot(hk1, hk2, hk3, layout=(1,3))
savefig(hk4, "K-histograms")

# Report the mean and standard deviation of each variable. 
varsk = ["A", "Y", "C"]
meansk = mean.([SimK.A, ℯ.^SimK.Y, SimK.C])
sdk = std.([SimK.A, ℯ.^SimK.Y, SimK.C])

# Plot the evolution of A, Y , and C over a 100 period stretch starting from period 1000
periods = 1000:1100
pk1 = plot(periods, SimK.A[periods], ylabel="A", label="")
pk2 = plot(periods, (ℯ.^SimK.Y)[periods], ylabel="Y", label="")
pk3 = plot(periods, SimK.C[periods], ylabel="C", xlabel="Period", label="")
pk4 = plot(pk1, pk2, pk3, layout=(3,1))
savefig(pk4, "K-evolutions")

# How do these mean values compare to the steady-state values calculated earlier?
# A comparison
Adiffk = meansk[1] - Aₛₛ
# Y comparison
Ydiffk = meansk[2] - μ
# C comparison
Cdiffk = meansk[3] - c(μ, Aₛₛ, Aₛₛ)
diffs = [Adiffk, Ydiffk, Cdiffk]
oldmeans = [Aₛₛ, μ, c(μ, Aₛₛ, Aₛₛ)]
stddiffs = [Adiffk/Aₛₛ*100, Ydiffk/μ*100, Cdiffk/c(μ, Aₛₛ, Aₛₛ)*100]
dfk = DataFrame(Variable = varsk, 
                Mean = meansk, 
                StdDev = sdk, 
                SteadStateValues = oldmeans, 
                DiffFromSS = diffs,
                StdDiff = stddiffs)
dfk


#=
The mean of income (Y) is pretty close to the given mean of 1 in the problem setup,
only about 2.4% higher.
The mean of assets (A) in the simulation is somewhat larger than the steady state value --
about 12 units above the steady state value, which is about 24.7% higher.
The mean of consumption (C) in the simulation is a bit lower than the
consumption implied by the mean income μ and the steady state asset level --
the mean from the simulation is about 11% below the steady state level.
=#



















#==============================================================================
                Part L
===============================================================================
Plot consumption as a function of A for several values of Y . Do this for the entire range of A on
your grid.

I think this has nothing to do with the simulation...
=#

# PLOT USING A' APPROXIMATION
bAp2 = PolyGetCoef(AA, YY, Ap2)
cL(A, lnY) = c(lnY, A, (PolyBasis(A, lnY) * bAp2)[1])
labelsL = "log(Y) = " .* string.(round.(GridlnY', digits=3))
linestylesL = [:solid :dot :solid :dot :solid :dot :solid]
pL1 = plot(reshape(cL.(AA, YY), nA, nY),
     label=labelsL,
     linestyle=linestylesL,
     legend=:none,
     xlabel="Assets at beginning of period",
     ylabel="Consumption")

# PLOT USING NEW C APPROXIMATION (and underlying A' approximation in cL())
bC = PolyGetCoef(AA, YY, cL.(AA, YY))
CL = PolyBasis(AA, YY)*bC
pL2 = plot(reshape(CL, nA, nY),
     label=labelsL,
     linestyle=linestylesL,
     legend=:topleft,
     xlabel="Assets at beginning of period",
     ylabel="")


pL3 = plot(pL1, pL2, layout=(1,2),
        title=[" "^30*"Consumption - A' vs C poly approximation" ""])
savefig(pL3, "L-consumption")


















#==============================================================================
                Part M
===============================================================================
Plot change in assets Y − C as a function A for several values of Y. 
Do this for the entire range of A on your grid.
=#

# PLOT USING A' APPROXIMATION
pM1 = plot(reshape(ℯ.^YY .- cL.(AA, YY), nA, nY),
     label=labelsL,
     linestyle=linestylesL,
     legend=:none,
     xlabel="Assets at beginning of period",
     ylabel="Y - C")

# PLOT USING NEW C APPROXIMATION (and underlying A' approximation in cL())
pM2 = plot(reshape(ℯ.^YY .- CL, nA, nY),
     label=labelsL,
     linestyle=linestylesL,
     legend=:bottomleft,
     xlabel="Assets at beginning of period",
     ylabel="")

pM3 = plot(pM1, pM2, layout=(1,2),
    title=[" "^20*"Change in Assets - A' vs C poly approximation" ""])
savefig(pM3, "M-changeinassets")

















#==============================================================================
                Part N
===============================================================================
Plot the marginal propensity to consume as a function of A for several values of Y . Do this for
the entire range of A on your grid. You can approximate the marginal propensity to consume as
the extra consumption in the period that results from a windfall gain of 1 unit of A. Does this plot
make economic sense? (Hint: It might not due to the limitations of the polynomial approximation
methods we are using in this problem set.)
=#

# PLOT USING A' APPROXIMATION (Swooshy MPC)
MPC1 = cL.(AA .+ 1, YY) .- cL.(AA, YY)
pN1 = plot(reshape(MPC1, nA, nY),
     label=labelsL,
     linestyle=linestylesL,
     legend=:none,
    #  title="Marginal Propensity to Consume based on optimal asset policy",
     xlabel="Assets at beginning of period",
     ylabel="Marginal Propensity to Consume")

# PLOT USING NEW C APPROXIMATION (Straight line MPC)
CN = PolyBasis(AA .+ 1, YY) * bC
MPC2 = PolyBasis(AA .+ 1, YY) * bC .- PolyBasis(AA, YY) * bC
pN2 = plot(reshape(MPC2, nA, nY),
     label=labelsL,
     linestyle=linestylesL,
     legend=:bottomright,
    #  title="Marginal Propensity to Consume",
     xlabel="Assets at beginning of period",
    #  ylabel="MPC"
     )

pN3 = plot(pN1, pN2, layout=(1,2),
            title=[" "^40*"MPC - A' vs C poly approximation" ""])
savefig(pN3, "N-mpc")


#=
I'm also wondering this... I can't wrap my head around why we would want to further approximate the consumption, after already approximating A'. It looks like the solutions use the polynomial approximation for C for parts L, M, and N.
My humble opinion is that it's less accurate to approximate C using polynomials; instead, using the optimal policy rule we already approximated to calculate the consumption at each point, we get the swooshy MPC curves that Sebastian posted earlier. I think these are actually more correct, and the straight lines just come from the fact that we're using a quadratic approximation and shifting it by one unit.

Plus, using the estimated coefficients for the assets to calculate consumption follows what McKay is doing on his website at the bottom of the Results! section:
https://alisdairmckay.com/Notes/NumericalCrashCourse/VFI.html#results
=#




















#==============================================================================
                Part O
===============================================================================
(Optional) Explore how the solution method runs into trouble if you try to increase α towards 1. As
you do this, you may want to vary the range of assets on the grid and also the polynomial basis. If
you consider higher order polynomials than 2nd order, it may be interesting for you to plot the value
function for a particular value of Yt as a function of At during intermediate steps in the value function
iteration. You may start seeing cases where the value function becomes slightly non-monotonic. You
can think about how this will lead the golden search algorithm to run into problems. (This is the
problem that we couldn’t get around easily in writing the problem.)
=#






















#=================== ARCHIVE ==================================================

#############################
    Max Bellman Functions
#############################
function MyMaxSingleBellman2(EV, Aₜ, lnYₜ; lbA=first(AA), ubA=last(AA), α=α)
    to_minimize(Ap) = -Bellman(EV, Aₜ, lnYₜ, Ap; α=α)
    # Want there to be >0 consumption, so put upper bound
    # at maximum A' that results in C>0, given lnYₜ, Aₜ
    upperA = min(ubA, max_Ap(lnYₜ, Aₜ; α=α) - 1e-3)
    # println("\nUB: $upperA,  LB: $lbA")
    out = optimize(to_minimize, lbA, upperA)
    maxBell = -1 * out.minimum
    maxA = out.minimizer
    return maxA, maxBell
end
function MyMaxBellman2(EV; α=α)
    # Define the function taking scalar Aₜ, lnY
    # MaxBellmanVector(Aₜ, lnY) = MyMaxSingleBellman2(EV, Aₜ, lnY; α=α)
    # Broadcast over this function
    # out = MaxBellmanVector.(AA, lnYY)
    out = MyMaxSingleBellman2.(EV, AA, YY)
    maxA = [x[1] for x in out]
    maxBell = [x[2] for x in out]
    println("var(maxA) = $(var(maxA)),  var(maxBell) = $(var(maxBell))")
    return maxBell, maxA
end
function MyBellmanIteration2()
    # initial guess of the coefficients of the polynomial approx to the value function (zero function)
    b = zeros(6)
    EV = Aₜ₊₁0 = zeros(size(AA))
    MAXIT = 2000
    Vlist, Alist, blist = [zeros(size(AA))], [zeros(size(AA))], [b]
    for it = 1:MAXIT

        println(size(EV), size(AA), size(YY))
        V, Aₜ₊₁ = MyMaxBellman2(EV; α=α)
        append!(Vlist, [V])
        append!(Alist, [Aₜ₊₁])

        # take the expectation of the value function from the perspective of the previous lnY
        # Need to reshape V into a 20 by 7 array where the rows correspond different levels
        # of assets and the columns correspond to different levels of income.
        # need to take the dot product of each row of the array with the appropriate column of the Markov chain transition matrix
        # So we are taking the expectation of the value function by multiplying each lnY level's value of the 
        # value function by the probability that we will still be in that level next period.
        EV = reshape(V, nA, nY) * GridPlnY
        EV = EV[:]

        # update our polynomial coefficients
        b = PolyGetCoef(AA, YY, EV)
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
# Plotting maximizations of the Bellman
x11(A, Y) = MyMaxSingleBellman(b_, A, Y)
x11out = x11.(AA, YY)
a11 = [x[1] for x in x11out]
v11 = [x[2] for x in x11out]
=#

# TESTING THE MyBellmanIteration Output
# @time Alist, Vlist, blist = MyBellmanIteration2()


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


#############################
    
#############################

=#
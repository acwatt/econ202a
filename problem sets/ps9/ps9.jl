#=
File: ps9.jl
Author: Aaron C Watt
Date: 2022-11-20
Purpose: Solve problems from 202AFall2022PS9.pdf

References:
https://alisdairmckay.com/Notes/NumericalCrashCourse/FuncApprox.html
https://alisdairmckay.com/Notes/NumericalCrashCourse/index.html

Expectations:
https://quantecon.github.io/Expectations.jl/dev/

Optimization:
https://julianlsolvers.github.io/Optim.jl/stable/
=#

#==============================================================================
                Packages
==============================================================================#
import Pkg
Pkg.activate(pwd())
try
    using Optim, Parameters, Plots, Revise, DataFrames, Interpolations, Distributions, Random
    using Optim: maximum, maximizer
    pyplot()
catch e
    Pkg.add(["Plots", "PyPlot", "Optim", "Parameters", "Revise", "DataFrames", "Interpolations", "Distributions", "Random"])
    using Optim, Parameters, Plots, Revise, DataFrames, Interpolations, Distributions, Random
    using Optim: maximum, maximizer
    pyplot()
end
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
No PolyBasis function 
=#

#==============================================================================
                Part C
===============================================================================
No PolyGetCoeff function
=#

#==============================================================================
                Part D
===============================================================================
Start your main Matlab program by reading in the parameter values into a structure.
=#
# Parameters Given
@consts begin
    γ = 2
    β = 0.98
    μ = 1
    bb = 0.4
    ρ = 0.9
    σ² = 0.05
    σ = σ²^(1/2)
    p = 0.05
    q = 0.25
    r = 0.01
end



# Labor Income
# Y(Yₜ₋₁, εₜ; ρ=ρ, μ=μ) = exp( (1 − ρ)*log(μ) + ρ*log(Yₜ₋₁) + εₜ )
lnY(lnYₜ₋₁, εₜ) = (1 - ρ) * log(μ) + ρ * lnYₜ₋₁ + εₜ










#==============================================================================
                Part E
===============================================================================
Create a grid on A and Ỹ . Feel free to use McKay’s tauchen function as needed. Use 7 grid points
for Ỹ and 1,000 grid points for A. (Please create the grid for Ỹ , not log Ỹ . Uniformity will make
grading easier.) Choose reasonable values for the size of the grid (i.e., min and max points for
each dimension). Note that the lower bound of the A grid in this case should be zero, since
the borrowing constraint prevents A from being negative. As in problem set 8, it is important
to choose a large enough value for the maximum of the A grid to get an accurate solution. But at
the same time, choosing something excessively large is inefficient. Some trial and error is probably
necessary here.
    A=K     Z=lnY
=#

# Create a grid for lnY
NY = 7;  # number of points in our grid for lnY
NstdY = 4;  # number of standard deviations to cover with the grid
# Note that we need to use log(μ) due to the formula used in tauchen()
GridlnY, GridPY = tauchen(NY, log(μ), ρ, σ, NstdY)
GridY = ℯ.^GridlnY
# tauchen() imported from Tauchen.jl

GridPY = GridPY'  # this is a 7 x 7 transition matrix for which the columns sum to 1
# the (i,j) element is the probability of moving from j to i.

# Create a grid from 0 to upper bound
GridA_upper = 200
GridA_lower = 0
NA = 1_000  # number of points in our grid for A
GridA = range(GridA_lower, GridA_upper, length=NA)

# Cartisian product of the grids, then decompose
AY = [(a, y) for y ∈ GridlnY for a ∈ GridA]
AA = [a for (a, y) ∈ AY]
YY = [y for (a, y) ∈ AY]




#==============================================================================
                Part F
===============================================================================
Write a Bellman matlab function for this problem. Since we have two value functions (Ve and
Vu), the best way to go here is to write two functions BellmanE and BellmanU, one for each value
function.
=#

# Utility function
U(C) = C^(1 - γ) / (1 - γ)

# This period's starting wealth
f(lnYₜ, Aₜ) = exp(lnYₜ) + Aₜ
# Savings this period based on next periods' assets
savings(Aₜ₊₁) = Aₜ₊₁/(1+r)

# Budget Constraint defines Cₜ(Yₜ, Aₜ, Aₜ₊₁) = income - savings
c(lnYₜ, Aₜ, Aₜ₊₁) = f(lnYₜ, Aₜ) - savings(Aₜ₊₁)
# Maximum A' could be for C>0, given Y and A
Aprime(lnYₜ, Aₜ, Cₜ) = (1+r) * (exp(lnYₜ) + Aₜ - Cₜ)
max_Ap(lnYₜ, Aₜ) = Aprime(lnYₜ, Aₜ, 0)

"""Labor Income: Y: employed income; X: 1 if employed, 0 o.w.; bb = unemployed income"""
Yfun(Y, X) = X ? Y : bb



"""Return interpolated values of EV at Aₜ₊₁ points given input EV vector on regular grid"""
function interpolate_EV(EV::AbstractArray, lnYₜ::AbstractArray, Aₜ₊₁::AbstractArray)
    # Convert EV to matrix for interpolation
    EVmat = reshape(EV, NA, NY)
    # Create interpolation function (based on regular grids 1:NA and 1:NY)
    Interp = interpolate(EVmat, BSpline(Cubic(Line(OnGrid()))))
    # Scale the inputs to match the actual grids
    sInterp = Interpolations.scale(Interp, GridA, GridlnY)
    # Interpolate on vectors of Aₜ₊₁ and lnYₜ
    EVinterp = sInterp.(Aₜ₊₁, lnYₜ)
    # The output will be multiplied by the markov transition matrix for Y later to EV(Aₜ₊₁, lnYₜ) → EV(Aₜ₊₁, lnYₜ₊₁)
    return EVinterp
end
function interpolate_EV(EV::AbstractArray, lnYₜ::Real, Aₜ₊₁::Real)
    # Convert EV to matrix for interpolation
    EVmat = reshape(EV, NA, NY)
    # Create interpolation function (based on regular grids 1:NA and 1:NY)
    Interp = interpolate(EVmat, BSpline(Cubic(Line(OnGrid()))))
    # Scale the inputs to match the actual grids
    sInterp = Interpolations.scale(Interp, GridA, GridlnY)
    # Interpolate on scalar Aₜ₊₁ and lnYₜ
    EVinterp = sInterp(Aₜ₊₁, lnYₜ)
    # The output will be multiplied by the markov transition matrix for Y later to EV(Aₜ₊₁, lnYₜ) → EV(Aₜ₊₁, lnYₜ₊₁)
    return EVinterp
end
function interpolate_EV(EV::AbstractArray)
    # Convert EV to matrix for interpolation
    EVmat = reshape(EV, NA, NY)
    # Create interpolation function (based on regular grids 1:NA and 1:NY)
    Interp = interpolate(EVmat, BSpline(Cubic(Line(OnGrid()))))
    # Scale the inputs to match the actual grids
    sInterp = Interpolations.scale(Interp, GridA, GridlnY)
    # Return interpolated function on scalar Aₜ₊₁ and lnYₜ
    EVinterp(Aₜ₊₁, lnYₜ) = sInterp(Aₜ₊₁, lnYₜ)
    return EVinterp
end


"""Vector Bellman function when Employed this period"""
function BellmanE(EVe::AbstractArray, EVu::AbstractArray, Aₜ::AbstractArray, lnYₜ::AbstractArray, Aₜ₊₁::AbstractArray)
    # EMPLOYED: vector A', lnY, EVe, EVu
    C = c.(lnYₜ, Aₜ, Aₜ₊₁)
    # Interpolate EVe and EVu at Aₜ₊₁, lnYₜ
    EVe2 = interpolate_EV(EVe, lnYₜ, Aₜ₊₁)
    EVu2 = interpolate_EV(EVu, lnYₜ, Aₜ₊₁)
    # P(emp | emp) = 1-p; P(unemp | emp) = P
    Ve = U.(C) .+ β*( (1-p)*EVe2 + p*EVu2 )
    return Ve
end
"""Scalar Bellman function when Employed this period.
    EVe, EVu are interpolation functions of (Aₜ₊₁, lnYₜ)
"""
function BellmanE(EVe::Function, EVu::Function, Aₜ::Real, lnYₜ::Real, Aₜ₊₁::Real)
    # EMPLOYED: vector A', lnY, EVe, EVu
    C = c(lnYₜ, Aₜ, Aₜ₊₁)
    # P(emp | emp) = 1-p; P(unemp | emp) = P
    Ve = U(C) + β*( (1-p)*EVe(Aₜ₊₁, lnYₜ) + p*EVu(Aₜ₊₁, lnYₜ) )
    return Ve
end


"""Vector Bellman function when Unemployed this period"""
function BellmanU(EVe::AbstractArray, EVu::AbstractArray, Aₜ::AbstractArray, Aₜ₊₁::AbstractArray)
    # UNEMPLOYED: vector A', lnY=bb, EVe, EVu
    C = c.(log(bb), Aₜ, Aₜ₊₁)
    # Interpolate EVe and EVu at Aₜ₊₁, lnYₜ
    lnYₜ = repeat([log(bb)], length(Aₜ))
    EVe2 = interpolate_EV(EVe, lnYₜ, Aₜ₊₁)
    EVu2 = interpolate_EV(EVu, lnYₜ, Aₜ₊₁)
    # P(emp | unemp) = q; P(unemp | unemp) = 1-q
    Vu = U.(C) .+ β*( q*EVe2 .+ (1-q)*EVu2 )
    return Vu
end
"""Scalar Bellman function when Unemployed this period.
    EVe, EVu are interpolation functions of (Aₜ₊₁, lnYₜ)
"""
function BellmanU(EVe::Function, EVu::Function, Aₜ::Real, Aₜ₊₁::Real)
    # UNEMPLOYED: vector A', lnY=bb, EVe, EVu
    C = c(log(bb), Aₜ, Aₜ₊₁)
    # P(emp | unemp) = q; P(unemp | unemp) = 1-q
    Vu = U(C) + β*( q*EVe(Aₜ₊₁, log(bb)) + (1-q)*EVu(Aₜ₊₁, log(bb)) )
    return Vu
end


#= testing the Bellman output


Ap_interp = min.(GridA[end], AA .+ 0.1)
Ap_interp = max.(0, Ap_interp)
BFe = BellmanE(zeros(size(AA)), zeros(size(AA)), AA, YY, Ap_interp)
BFu = BellmanU(zeros(size(AA)), zeros(size(AA)), AA, Ap_interp)

EVeF(Aₜ₊₁) = interpolate_EV(zeros(size(AA)), Aₜ₊₁)
EVuF(Aₜ₊₁) = interpolate_EV(zeros(size(AA)), Aₜ₊₁)
BFe2 = BellmanE.(EVeF, EVuF, AA, YY, Ap_interp)
BFu2 = BellmanU(EVeF, EVuF, AA, Ap_interp)


angle0 = (-45,30)
p1_ = surface(AA, YY, BFe, xlabel="A'", ylabel="lnY", zlabel="Bellman", camera=angle0)
surface(YY, AA, BFe, xlabel="lnY", ylabel="A'", zlabel="Bellman", camera=angle0)
p2_ = surface(AA, YY, BFu, xlabel="A'", ylabel="lnY", zlabel="Bellman", camera=angle0)
surface(YY, AA, BFu, xlabel="lnY", ylabel="A'", zlabel="Bellman", camera=angle0)

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
Write a version of McKay’s MaxBellman function for this problem. Again, you will have two
functions MaxBellmanE and MaxBellmanU. These should be quite similar to the analogous function
in problem set 8. The main difference is that you will be passing the entire grids on assets and income
to the Bellman function. (So, you will have something like grid.AA as an argument that you pass
to Bellman as opposed to grid.AA(I) in problem set 8 (and the same for the grid on income).)
=#



#*****************************************************************************
#*****************************************************************************
# The below functions converge to the same value and policy functions as the
# the translation of McKay's code, but the following functions use Optim.jl's
# optimizer to find the maximizing A' instead of the golden search method.
#*****************************************************************************
#*****************************************************************************

function MyMaxSingleBellmanE(EVe, EVu, Aₜ, lnYₜ)
    # Define a univariate function to maximize over Aₜ₊₁
    to_maximize(Aₜ₊₁) = BellmanE(EVe, EVu, Aₜ, lnYₜ, Aₜ₊₁)
    # Want there to be >0 consumption, so put upper bound
    # at maximum A' that results in C>0, given lnYₜ, Aₜ
    upperA = min(GridA_upper, max_Ap(lnYₜ, Aₜ) - 1e-3)
    # Find the maximizing Aₜ₊₁ for this point in the Aₜ, lnYₜ grid
    out = maximize(to_maximize, GridA_lower, upperA)
    V = maximum(out)
    Aₜ₊₁ = maximizer(out)
    return Aₜ₊₁, V
end
function MyMaxSingleBellmanU(EVe, EVu, Aₜ)
    # Define a univariate function to maximize over Aₜ₊₁
    to_maximize(Aₜ₊₁) = BellmanU(EVe, EVu, Aₜ, Aₜ₊₁)
    # Want there to be >0 consumption, so put upper bound
    # at maximum A' that results in C>0, given lnYₜ=bb, Aₜ
    upperA = min(GridA_upper, max_Ap(log(bb), Aₜ) - 1e-3)
    # Find the maximizing Aₜ₊₁ for Aₜ, lnYₜ=bb
    out = maximize(to_maximize, GridA_lower, upperA)
    V = maximum(out)
    Aₜ₊₁ = maximizer(out)
    return Aₜ₊₁, V
end

function MyMaxBellmanE(EVe::AbstractArray, EVu::AbstractArray)
    # Create interpolation functions (functions of lnYₜ, Aₜ₊₁)
    EVefun = interpolate_EV(EVe)
    EVufun = interpolate_EV(EVu)
    # Define the function taking scalar Aₜ, lnY
    MaxBellmanVector(Aₜ, lnYₜ) = MyMaxSingleBellmanE(EVefun, EVufun, Aₜ, lnYₜ)
    # Broadcast over this function
    out = MaxBellmanVector.(AA, YY)
    maxA = [x[1] for x in out]
    maxBell = [x[2] for x in out]
    return maxBell, maxA
end
function MyMaxBellmanU(EVe::AbstractArray, EVu::AbstractArray)
    # Create interpolation functions (functions of lnYₜ, Aₜ₊₁)
    EVefun = interpolate_EV(EVe)
    EVufun = interpolate_EV(EVu)
    # Define the function taking scalar Aₜ, lnY
    MaxBellmanVector(Aₜ) = MyMaxSingleBellmanU(EVefun, EVufun, Aₜ)
    # Broadcast over this function
    out = MaxBellmanVector.(AA)
    maxA = [x[1] for x in out]
    maxBell = [x[2] for x in out]
    return maxBell, maxA
end













#==============================================================================
                Part H
===============================================================================
Write the value function iteration for-loop for this problem.
=#

"""Update value function and A' lists of vectors with new vectors"""
function update_lists!(Velist, Vulist, Aelist, Aulist, Ve, Vu, Ae, Au)
    append!(Velist, [Ve])
    append!(Vulist, [Vu])
    append!(Aelist, [Ae])
    append!(Aulist, [Au])
end
function update_lists!(Velist, Vulist, Ve, Vu)
    append!(Velist, [Ve])
    append!(Vulist, [Vu])
end

"""Iterate over EV and Ap vectors to converge on the value function and Ap policy rule"""
function MyBellmanIteration(; verbose=true)
    # initial guess of the value functions (zero function)
    MAXIT = 2_000
    Velist, Vulist, Aelist, Aulist = [zeros(size(AA))], [zeros(size(AA))], [zeros(size(AA))], [zeros(size(AA))]
    for it = 1:MAXIT
        Ve, Ape = MyMaxBellmanE(Velist[end], Vulist[end])
        Vu, Apu = MyMaxBellmanU(Velist[end], Vulist[end])

        # take the expectation of the value function from the perspective of the previous A
        # Need to reshape V into a 100x7 array where the rows correspond different levels
        # of assets and the columns correspond to different levels of income.
        # need to take the dot product of each row of the array with the appropriate column of the Markov chain transition matrix
        EVe = reshape(Ve, NA, NY) * GridPY
        EVu = reshape(Vu, NA, NY) * GridPY

        # update our value functions
        update_lists!(Velist, Vulist, Aelist, Aulist, EVe[:], EVu[:], Ape, Apu)

        # see how much our policy rules and value functions have changed
        Aetest = maximum(abs.(Aelist[end] - Aelist[end-1]))
        Autest = maximum(abs.(Aulist[end] - Aulist[end-1]))
        Vetest = maximum(abs.(Velist[end] - Velist[end-1]))
        Vutest = maximum(abs.(Vulist[end] - Vulist[end-1]))
        
        if it % 50 == 0
            verbose ? println("iteration $it, Vetest = $Vetest, Vutest = $Vutest, Aetest = $Aetest, Autest = $Autest") : nothing
        end
        if max(Aetest, Autest, Vetest, Vutest) < 1e-5
            println("\nCONVERGED -- final iteration tests:")
            println("iteration $it, Vetest = $Vetest, Vutest = $Vutest, Aetest = $Aetest, Autest = $Autest")
            break
        end

        it == MAXIT ? println("\nMAX ITERATIONS REACHED ($MAXIT)") : nothing
    end

    return Velist, Vulist, Aelist, Aulist
end

RUNH = false
if RUNH
    outH = @time MyBellmanIteration();
    VeH = outH[1][end]; VuH = outH[2][end]; ApeH = outH[3][end]; ApuH = outH[4][end];
end

PLOTH = false
if PLOTH
    angle0 = (-45,30)
    plotargs = (camera=(-45, 20), xlabel="Y", ylabel="A",
                legend=:none, aspect_ratio=[1,1,2])
    pVe_H = surface(exp.(YY), AA, VeH, zlabel="Ve"; plotargs...)
    pVu_H = surface(exp.(YY), AA, VuH, zlabel="Vu"; plotargs...)
    pApe_H = surface(exp.(YY), AA, ApeH, zlabel="Ape"; plotargs...)
    pApu_H = surface(exp.(YY), AA, ApuH, zlabel="Apu"; plotargs...)
    pCe_H = surface(exp.(YY), AA, c.(YY, AA, ApeH), zlabel="Ce"; plotargs...)
    pCu_H = surface(exp.(YY), AA, c.(log(bb), AA, ApuH), zlabel="Cu"; plotargs...)

    pH1 = plot(pVe_H, pVu_H, pApe_H, pApu_H, pCe_H, pCu_H, 
        layout=(3,2), size=(800, 1600))
    savefig(pH1, "H-all_$GridA_upper")
end



#= INTERPOLATE OPTIMAL ASSET POLICY FOR POINTS NOT ON GRID
# McKay's code
bKp =  PolyGetCoef(Grid.KK,Grid.ZZ,Kp);
Kp2903 = PolyBasis(29,0.03) * bKp
C2903 = f(Par,29,0.03) - Kp2903

# Translation for point A=10, lnY=0.1
bAp2 = PolyGetCoef(AA, YY, Ap2)
Ap1001 = PolyBasis(10, 0.1) * bAp2
C1001 = c(0.1, 10, Ap1001[1])
=#













#==============================================================================
                Part I
===============================================================================
(Optional) Implement the Howard acceleration for this problem. Report the speed improvement that
you are able to achieve. (You will need to experiment with the number of iterations to figure out
what works well in terms of giving a speed improvement.)
=#


"""Iterate faster over the value function vectors to converge on the value function using Howard acceleration."""
function MyFasterBellmanIteration(; inner_mod = 32, verbose=false)
    println("\nStarting MyFasterBellmanIteration with inner_mod = $inner_mod")
    # initial guess of the value functions (zero function)
    MAXIT = 20_000; inner_it = 0
    Velist, Vulist, Aelist, Aulist = [zeros(size(AA))], [zeros(size(AA))], [zeros(size(AA))], [zeros(size(AA))]
    for it = 1:MAXIT
        # Every inner_mod iterations, get the maximizing A'
        if it % round(inner_mod) == 1
            Ve, Ape = MyMaxBellmanE(Velist[end], Vulist[end])
            Vu, Apu = MyMaxBellmanU(Velist[end], Vulist[end])
            update_lists!(Velist, Vulist, Aelist, Aulist, Ve, Vu, Ape, Apu)
            inner_it += 1
        else
            # V = bellman(EV)
            Ve = BellmanE(Velist[end], Vulist[end], AA, YY, Aelist[end])
            Vu = BellmanU(Velist[end], Vulist[end], AA, Aulist[end])
        end

        # EV = reshape(V, NA, NY) * GridPY
        EVe = reshape(Ve, NA, NY) * GridPY
        EVu = reshape(Vu, NA, NY) * GridPY

        # append!(EVlist, EV[:])
        update_lists!(Velist, Vulist, EVe[:], EVu[:])

        # see how much our policy rules and value functions have changed
        Aetest = maximum(abs.(Aelist[end] - Aelist[end-1]))
        Autest = maximum(abs.(Aulist[end] - Aulist[end-1]))
        Vetest = maximum(abs.(Velist[end] - Velist[end-1]))
        Vutest = maximum(abs.(Vulist[end] - Vulist[end-1]))
        
        if it % 50 == 0
            verbose ? println("iteration $it, Vetest = $Vetest, Vutest = $Vutest, Aetest = $Aetest, Autest = $Autest") : nothing
        end
        if max(Aetest, Autest, Vetest, Vutest) < 1e-5
            verbose ? println("\nCONVERGED in $it iterations, $inner_it maximization iterations -- final iteration tests:") : nothing
            verbose ? println("iteration $it, Vetest = $Vetest, Vutest = $Vutest, Aetest = $Aetest, Autest = $Autest") : nothing
            break
        end

        it == MAXIT ? println("\nMAX ITERATIONS REACHED ($MAXIT)") : nothing
    end

    return Velist, Vulist, Aelist, Aulist
end

# warm up function (precompile)
outI = @time MyFasterBellmanIteration();
VeI = outI[1][end]; VuI = outI[2][end]; ApeI = outI[3][end]; ApuI = outI[4][end];


# Find which modulo for the bellman A' maximization results in shortest time
if 1==0
    mods = 10:100
    f(x::Real) = @elapsed @time MyFasterBellmanIteration(inner_mod=x; verbose=false);
    f(x::AbstractArray) = @elapsed MyFasterBellmanIteration(inner_mod=x[1]; verbose=false);
    times = f.(mods)
    mintime0, minidx = findmin(times)
    minmod = mods[minidx]

    # Compare to unaccelerated iteration
    mintime1 = @elapsed MyBellmanIteration();
    multiplier = round(mintime1 / mintime0, digits=2)
    println("Howard acceleration with mod $minmod resulted in $multiplier times faster convergence")

    plot(mods,times, xlabel="inner modulo", ylabel="time to converge (s)")
end

# out = maximize(to_maximize, GridA_lower, upperA)
# outI = optimize(f, 10, 100)
# This doesn't stop running. It eventually gets to a point and just keeps evaluating the function



















#==============================================================================
                Part J
===============================================================================
Adapt McKay’s Simulate function for this problem.
=#

struct SimReturn
    Ap # Assets at beginning of next period
    A  # Assets at beginning of period
    Ytilde # Employment Income
    Y  # Income = labor income*X + unemp income *(1-X)
    C  # Consumption
    X  # employment
end


# Define probability distributions for employment next period
distE = Binomial(1, 1-p)  # if emp this period, 1-p Prob of being emp next period
distU = Binomial(1, q)    # if unemp this period, q Prob of being emp next period

"""Return employment status next period X' based on employment status this period X"""
function employment_next(X)
    Xp = X==1 ? rand(distE, 1)[1] : rand(distU, 1)[1]
    return Xp
end


"""
Sim = Simulate(ApeI, ApuI, Mode, T)
    Simulates the model.
    Inputs:
    ApeI      Estimated A' emplyed policy rule (interpolated function of Aₜ, lnYₜ)
    ApuI      Estimated A' unemployed policy rule (interpolated function of Aₜ, lnYₜ)
    Mode      Mode = 'random' -> draw shocks
              Mode = 'irf'    -> impulse response function
    T         # of periods to simulate
"""
function Simulate(Ape, Apu, Mode, T)
    Random.seed!(123);
    A = zeros(T)  # Assets
    Ytilde = zeros(T)  # Employment income
    Y = zeros(T)  # Realized Income (employed or not)
    X = zeros(T)  # Employment status {0,1}
    A[1] = mean(GridA)
    X[1] = 1  # Start employed

    if Mode == "irf"
        Ytilde[1] = σ
        ε = zeros(T)
    elseif Mode == "random"
        Ytilde[1] = log(μ)
        ε = σ * randn(T)
    else
        throw("Unrecognized Mode $Mode in Simulate()");
    end

    for t ∈ 2:T
        # println("t=$t, A = $(A[t-1]), Y = $(Y[t-1]), X = $(X[t-1])")
        # Employed?
        X[t] = employment_next(X[t-1])
        # Labor income
        Ytilde[t] = lnY(Ytilde[t-1], ε[t])
        # Realized income (considering unemployment)
        Y[t] = X[t]*Ytilde[t] + (1-X[t])*log(bb)
        # Realized Assets
        ApFunc = X[t]==1 ? Ape : Apu
        A[t] = max(ApFunc(A[t-1], Y[t-1]), 0)  # Aₜ ≥ 0
    end
    
    # Compute quantities from state variables
    Ti = 2:(T-1)
    Ap = A[Ti .+ 1]
    A = A[Ti]
    Ytilde = Ytilde[Ti]
    Y = Y[Ti]
    C = c.(Y, A, Ap)

    return SimReturn(Ap, A, Ytilde, Y, C, X)
end














#==============================================================================
                Part K
===============================================================================
Using your new Simulate function, produce a 10,000 period simulation of the evolution of A, Ỹ , C,
and employment status for a single household. Report a histogram of A, Y , and C. Report the mean
and standard deviation of each variable. Plot the evolution of A, Y , and C over a 100 period stretch
starting from period 1000. (Note that I am asking you to report Y , not Ỹ .)
=#
# Compare output from different value function iteration methods
# Ape_diff = maximum(abs.(ApeI .- ApeH))
# Apu_diff = maximum(abs.(ApuI .- ApuH))
# Ap_diff = maximum(abs.(ApeI .- ApuI))
# Converged within 1e-5 tolerance. let's use ApeI and ApuI

# Generate interpolated A' functions of Ape and Apu
ApeFunc(A, lnY) = interpolate_EV(ApeI)(A, lnY)
ApuFunc(A, lnY) = interpolate_EV(ApuI)(A, lnY)
# surface(exp.(YY), AA, ApeFunc.(AA, YY), zlabel="ApeFunc"; plotargs...)
# surface(exp.(YY), AA, ApuFunc.(AA, YY), zlabel="ApuFunc"; plotargs...)

# Simulate
nPeriods = 10_000
@time SimK = Simulate(ApeFunc, ApuFunc, "random", nPeriods);

# Report the mean and standard deviation of each variable. 
varsk = ["A", "Y", "C"]
meansk = mean.([SimK.A, ℯ.^SimK.Y, SimK.C])
sdk = std.([SimK.A, ℯ.^SimK.Y, SimK.C])

# Report a histogram of A, Y , and C. (Note: Yt not log Yt .) 
hk1 = histogram(SimK.A, title="A", label="") #, xlims=(61,64)
annotate!(hk1, [(0.7*maximum(SimK.A), 800, "Mean=$(round(meansk[1], digits=2))"),
                (0.7*maximum(SimK.A), 700, " StD=$(round(sdk[1], digits=2))")])
hk2 = histogram(ℯ.^SimK.Y, title="Y", label="")
annotate!(hk2, [(0.7*maximum(ℯ.^SimK.Y), 1500, "Mean=$(round(meansk[2], digits=2))"),
                (0.7*maximum(ℯ.^SimK.Y), 1300, " StD=$(round(sdk[2], digits=2))")])
hk3 = histogram(SimK.C, title="C", label="") #, xlims=(31,35)
annotate!(hk3, [(0.7*maximum(SimK.C), 800, "Mean=$(round(meansk[3], digits=2))"),
                (0.7*maximum(SimK.C), 700, " StD=$(round(sdk[3], digits=2))")])
hk4 = plot(hk1, hk2, hk3, layout=(1,3))
savefig(hk4, "K-histograms")


# Plot the evolution of A, Y , and C over a 100 period stretch starting from period 1000
periods = 1000:1100
pk1 = plot(periods, SimK.A[periods], ylabel="A", label="")
pk2 = plot(periods, (ℯ.^SimK.Y)[periods], ylabel="Y", label="")
hline!(pk2, [bb], label="unemp. income")
pk3 = plot(periods, SimK.C[periods], ylabel="C", xlabel="Period", label="")
pk4 = plot(pk1, pk2, pk3, layout=(3,1))
savefig(pk4, "K-evolutions")






















#==============================================================================
                Part L
===============================================================================
Plot consumption as a function of A for several values of Ỹ. Do this for the a 
range of values for A that encompasses most of the mass in the histogram you 
report in part K. On the same figure, also plot consumption when unemployed as 
a function of A. Do this for the average value of Ỹ .
=#

# hLY1 = histogram(ℯ.^SimK.Ytilde, title="ỹ", label="")
# hLY2 = histogram(ℯ.^SimK.Y, title="y", label="")
# hLY3 = plot(hLY1, hLY2, layout=(1,2))

# Generate values of Ỹ to plot for
YmeanL = mean(ℯ.^SimK.Ytilde); YsdL = std(ℯ.^SimK.Ytilde);
Yvals = range(GridY[1], YmeanL + 2*YsdL, 7)

# Generate Consumption at values
cLe(A, lnY) = c(lnY, A, ApeFunc(A, lnY))  # employed
cLu(A) = c(log(bb), A, ApuFunc(A, log(bb)))  # unemployed
CL = [[cLe(a, log(y)) for a in GridA, y in Yvals] cLu.(GridA)]

# PLOT USING A' APPROXIMATION
labelsL = "(Employed)   Y = " .* string.(round.(Yvals, digits=3)')
labelsL = [labelsL ["(Unemployed) Y = $bb"]]
linestylesL = [:solid :dot :solid :dot :solid :dot :solid :solid]
linecolorsL = [repeat([:auto], outer=(1,7)) [:grey]]
linewidthsL = [repeat([:auto], outer=(1,7)) [2]]
pL1 = plot(GridA, CL,
     label=labelsL,
     linestyle=linestylesL,
     linecolor=linecolorsL,
     linewidth=linewidthsL,
     legend=:bottomright,
     xlims=(0,150),
     xlabel="Assets at beginning of period",
     ylabel="Consumption",
     title="Consumption vs Assets using interpolated A' functions")
savefig(pL1, "L-consumption")




















#==============================================================================
                Part M
===============================================================================
Plot change in assets Y − C when employed as a function A for several values of Ỹ . Do this
for the same range of values for A as part L. On the same figure, also plot change in assets when
unemployed as a function of A. Do this for the average value of Ỹ .
=#

# PLOT USING A' APPROXIMATION
YM = [reshape(ℯ.^YY, NA, NY) repeat([bb], NA)]

pM1 = plot(GridA, YM .- CL,
     label=labelsL,
     linestyle=linestylesL,
     linecolor=linecolorsL,
     linewidth=linewidthsL,
     legend=:topright,
     xlims=(0,150),
     xlabel="Assets at beginning of period",
     ylabel="Y - C", title="Change in Assets")
savefig(pM1, "M-changeinassets")

















#==============================================================================
                Part N
===============================================================================
Plot the marginal propensity to consume when employed as a function of A for 
several values of Ỹ. On the same figure, also plot the marginal propensity to
consume when unemployed as a function of A (again for the average value of Ỹ ).
Do this for the entire range of A on your grid. You can approximate the marginal
propensity to consume as the extra consumption in the period that results from
a windfall gain of 1 unit of A.
=#

# Get index of points that will be inside the domain after shifting up by one
idxGridA_inside = findall(GridA_lower .<= GridA .+1 .<= GridA_upper)
# Filter previous consumption matrix to just those rows
CL_inside = CL[idxGridA_inside, :]
# Generate new consumption matrix for the shifted up grid
GridA_inside = GridA[idxGridA_inside]
GridA_shift = GridA_inside .+ 1
CN = [[cLe(a, log(y)) for a in GridA_shift, y in Yvals] cLu.(GridA_shift)]
# PLOT USING A' APPROXIMATION on smaller subset of GridA
MPC = CN .- CL_inside
pN1 = plot(GridA_inside, MPC,
     label=labelsL,
     linestyle=linestylesL,
     linecolor=linecolorsL,
     linewidth=linewidthsL,
     legend=:topright,
     xlims=(0,150), ylims=(0, maximum(MPC)),
     title="Marginal Propensity to Consume from optimal asset policy",
     xlabel="Assets at beginning of period",
     ylabel="Marginal Propensity to Consume")
savefig(pN1, "N-mpc")


#=
I'm also wondering this... I can't wrap my head around why we would want to further approximate the consumption, after already approximating A'. It looks like the solutions use the polynomial approximation for C for parts L, M, and N.
My humble opinion is that it's less accurate to approximate C using polynomials; instead, using the optimal policy rule we already approximated to calculate the consumption at each point, we get the swooshy MPC curves that Sebastian posted earlier. I think these are actually more correct, and the straight lines just come from the fact that we're using a quadratic approximation and shifting it by one unit.

Plus, using the estimated coefficients for the assets to calculate consumption follows what McKay is doing on his website at the bottom of the Results! section:
https://alisdairmckay.com/Notes/NumericalCrashCourse/VFI.html#results
=#




















#==============================================================================
                Part O
===============================================================================
No need to explore how accuracy depends on polynomial basis.
=#






















#=================== ARCHIVE ==================================================

#############################
    Bellman Functions
#############################
#=

"""
V = BellmanE(EV, Aₜ, lnYₜ, Aₜ₊₁; α=α)
  Evaluate the RHS of the Bellman equation if employed

    Inputs
    EV     6 x 1 coefficients in polynomial for E[ V(A',lnY') | lnY ]
    Aₜ     n-vector of current assets A
    lnYₜ   n-vector of current labor income
    Aₜ₊₁   n-vector of this period's savings (A')
    α     scalar savings technology parameter

    Output
    V     n-vector of value function
"""
function BellmanE(EV::Real, Aₜ::Real, lnYₜ::Real, Aₜ₊₁::Real)
    # EMPLOYED: Scalar A', lnY, EV
    C = c(lnYₜ, Aₜ, Aₜ₊₁)
    u = U(C)
    V = u + β*EV
    return V
end
function BellmanU(EV::Real, Aₜ::Real, Aₜ₊₁::Real)
    # UNEMPLOYED: Scalar A', EV -- bb = unemployment income
    C = c(bb, Aₜ, Aₜ₊₁)
    u = U(C)
    V = u + β*EV
    return V
end
function BellmanE(EV::AbstractArray, Aₜ::AbstractArray, lnYₜ::AbstractArray, Aₜ₊₁::AbstractArray)
    # EMPLOYED: Vector A', lnY, EV
    V = BellmanE.(EV, Aₜ, lnYₜ, Aₜ₊₁)
    return V
end

"""EV = value function; Aₜ, Aₜ₊₁ = assets; lnYₜ = income if employed; Xₜ = employed status"""
function Bellman(EVe::Real, EVu::Real, Aₜ::Real, lnYₜ::Real, Aₜ₊₁::Real, Xₜ::Real)
    income = Yfun(lnYₜ, Xₜ)
    C = c(income, Aₜ, Aₜ₊₁)
    u = U(C)
    probEmp = Xₜ ? 1-p : q    # P( employed next period  | employment status this period)
    probUnp = Xₜ ? p   : 1-q  # P(unemployed next period | employment status this period)
    V = u + β*(probEmp*EVe + probUnp*EVu)
    return V
end

=#






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
        EV = reshape(V, nA, NY) * GridPY
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
using Plots
gr()

root = dirname(@__FILE__)
file = "example.csv"
filepath = joinpath(root, file)

"""
PolyBasis(K,Z)
    Polynomial basis functions.  Using 2nd order polynomial
    inputs
    K    n x 1   points for K
    Z    n x 1   points for Z (or scalar)
    outputs
    B    n x 6   array of basis functions: 1, K, Z, K^2, K*Z, Z^2
"""
function PolyBasis(X,Y)
    Yb = Y.*ones(size(X));
    B = [ones(size(X)) X Yb X.^2 X.*Yb Yb.^2];
    return B
end
function PolyBasis(X)
    B = [ones(size(X)) X X.^2]
    return B
end

function gen_vectors(f, start, stop, n)
    A = range(start, stop, n)
    X = Y = Z = []
    for x in A for y in A
        append!(X, x)
        append!(Y, y)
        append!(Z, f(x,y))
    end end
    return X,Y,Z
end
function gen_vectors(f, start, stop, n)
    A = range(start, stop, n)
    X = Z = []
    for x in A
        append!(X, x)
        append!(Z, f(x))
    end
    return X,Z
end



"""
PolyGetCoef(Grid,Y)
    Fits the polynomial from PolyBasis to the function(s) in column(s) of Y.
    inputs
    K    n x 1   points for K
    Z    n x 1   points for Z
    Y    n x 1   valies for function at (K,Z)

    outputs
    b    6 x 1   basis coefficients
"""
function PolyGetCoef(X,Y,Z)
    B = PolyBasis(X,Y)
    b = B \ Z;
    return b
end
function PolyGetCoef(X,Z)
    B = PolyBasis(X)
    b = B \ Z;
    return b
end




f(x) = x<2 ? 0 : (x-2)^(1/2)
f(x,y) = log(x+y)

function gen_2d_coef(n)
    START=0.1; STOP=2.0
    X = Y = range(START,STOP,n)
    XY = [(x,y) for x ∈ X for y ∈ Y]
    XX = [x for (x,y) ∈ XY]
    YY = [y for (x,y) ∈ XY]
    Z = [log(x+y) for (x,y) ∈ XY]
    b = PolyGetCoef(XX,YY,Z)
    return b
end
function gen_1d_coef(n)
    START=0.1; STOP=4.0
    X = range(START,STOP,n)
    Z = [f(x) for x ∈ X]
    b = PolyGetCoef(X,Z)
    return b
end



function gen_2d_extrapolation(n, b; START=0.1, STOP=2.5)
    X = range(START,STOP,n)
    Y = range(START,STOP,n)
    XY = [(x,y) for x ∈ X for y ∈ Y]
    XX = [x for (x,y) ∈ XY]
    YY = [y for (x,y) ∈ XY]
    Z = [log(x+y) for (x,y) ∈ XY]
    B = PolyBasis(XX,YY)
    Zhat = B*b
    ε = Z - Zhat
    norm = maximum(abs.(ε))
    return (XX,YY), Z, Zhat, ε, norm
end
function gen_1d_extrapolation(n, b; START=0.1, STOP=4.5)
    X = range(START,STOP,n)
    Z = [f(x) for x ∈ X]
    B = PolyBasis(X)
    Zhat = B*b
    ε = Z - Zhat
    norm = maximum(abs.(ε))
    return (X,), Z, Zhat, ε, norm
end


function plot_stuff(X, Z, Zhat, ε, n)
    println("plotting lines for n=$n")
    plot(X, Z, title="True and Extrapolated Approximation (n=$n)")
    display(plot!(X, Zhat))

    display(plot(X, ε, title="Approximation Error (n=$n)"))

end
function plot_stuff(X, Y, Z, Zhat, ε, n)
    println("plotting surface for n=$n")
    surface(X, Y, Z, title="True and Extrapolated Approximation (n=$n)")
    display(surface!(X, Y, Zhat))

    display(surface(X, Y, ε, title="Approximation Error (n=$n)"))
end



for dimensions in 1:2
    for n ∈ [5, 15, 35]
        println("n=$n, dim=$dimensions")
        if dimensions == 1  # select 1 dimensional problem
            gen_coef = gen_1d_coef
            gen_extrap = gen_1d_extrapolation
            stop = 5
        elseif dimensions == 2  # select 2 dimensional problem
            gen_coef = gen_2d_coef
            gen_extrap = gen_2d_extrapolation
            stop = 2.5
        end
        # Generate polynomial coefficients and return true function values
        b = gen_coef(n)
        # Generate extrapolation
        X, Z, Zhat, ε, norm = gen_extrap(n, b; STOP=stop)
        # Plot true function, approximate, and error
        plot_stuff(X..., Z, Zhat, ε, n)
    end
end



# Surfaces for f(x,y)=log(x+y)



# Lines for f(x)=f()


# Create dataframe
# add each 












# n=5; START=0.1; STOP=2.0
# X = range(START,STOP,n)
# Y = range(START,STOP,n)
# XY = [(x,y) for x ∈ X for y ∈ Y]
# XX = [x for (x,y) ∈ XY]
# YY = [y for (x,y) ∈ XY]
# B=PolyBasis(XX,YY)
# Z1true = [log(x+y) for (x,y) ∈ XY]
# b1=PolyGetCoef(XX,YY,Z1true)
# Z1approx = B*b1
# ε1 = Z1true - Z1approx
# norm1 = maximum(abs.(ε1))



# n=49; START=0.1; STOP=2.5;
# X = range(START,STOP,n)
# Y = range(START,STOP,n)
# XY = [(x,y) for x ∈ X for y ∈ Y]
# XX = [x for (x,y) ∈ XY]
# YY = [y for (x,y) ∈ XY]
# Zeval_true = [log(x+y) for (x,y) ∈ XY]

# B = PolyBasis(XX,YY)
# Zeval_approx = B*b1
# ε2 = Zeval_true - Zeval_approx
# [Zeval_true Zeval_approx ε2]
# norm2 = maximum(abs.(ε2))



# using Plots
# plotly()
# surface(XX, YY, Zeval_true)
# surface!(XX, YY, Zeval_approx)

# surface(XX,YY,ε)






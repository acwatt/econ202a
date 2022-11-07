### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 50b76a1d-8cc2-4fa0-892a-ba19e345e5cf
# ╠═╡ show_logs = false
begin
	using Pkg; Pkg.add(["Plots", "DataFrames", "PyPlot"])
	using Plots
	gr()
	using DataFrames
	DIGITS = 4  # rounding to DIGITS for display
end;

# ╔═╡ b52f0010-5be8-11ed-04f2-63ec2d2c7f26
md"""
# Problem Set 7
Working with: Aidan Wang, Tak-Huen Chau, John Kadlick
"""

# ╔═╡ cbf11ac8-b1a9-4522-b09e-b926be6d382d
md"## Functions"

# ╔═╡ bca3746e-7066-48a7-91cf-d149d0c04a45
"""
PolyBasis(K,Z)
    Polynomial basis functions.  Using 2nd order polynomial
    inputs
    K    n x 1   points for K
    Z    n x 1   points for Z (or scalar)
    outputs
    B    n x 6   array of basis functions: 1, K, Z, K^2, K*Z, Z^2

PolyBasis(X)
	PolyBasis for the 1-d problem
"""
function PolyBasis(X,Y)
    Yb = Y.*ones(size(X));
    B = [ones(size(X)) X Yb X.^2 X.*Yb Yb.^2];
    return B
end

# ╔═╡ 742ee234-a942-4a71-b739-e5a734392138
function PolyBasis(X)
    B = [ones(size(X)) X X.^2]
    return B
end;

# ╔═╡ 76f05bd0-669b-4032-b2d2-5b24f1f1b821
"""Generate X,Y,Z for the 2-d problem"""
function gen_2d_vectors(f, start, stop, n)
    A = range(start, stop, n)
    X = Y = Z = []
    for x in A for y in A
        append!(X, x)
        append!(Y, y)
        append!(Z, f(x,y))
    end end
    return X,Y,Z
end

# ╔═╡ c207865e-dfcb-4312-9cd6-6a5719b1c206
"""Generate X,Z for the 1-d problem"""
function gen_1d_vectors(f, start, stop, n)
    A = range(start, stop, n)
    X = Z = []
    for x in A
        append!(X, x)
        append!(Z, f(x))
    end
    return X,Z
end

# ╔═╡ 91a6674f-951b-4e4d-bcbc-429e50a508fb
"""
PolyGetCoef(X,Y,Z)

Fits the polynomial from PolyBasis to the function(s) in column(s) of Z.

inputs

    K    n x 1   points for K
    Z    n x 1   points for Z
    Y    n x 1   values for function at (X,Y)

outputs

    b    6 x 1   basis coefficients
"""
function PolyGetCoef(X,Y,Z)
    B = PolyBasis(X,Y)
    b = B \ Z;
    return b
end

# ╔═╡ 25f8282b-a45d-4d2e-8cef-90206baa3de1
function PolyGetCoef(X,Z)
    B = PolyBasis(X)
    b = B \ Z;
    return b
end;

# ╔═╡ 9f252393-3eb9-4749-8df8-381ab5c369bd
"""
	f(x)

f(x) and f(x,y) for the problem

f(x) = (x-2)^(1/2) if x>=2, else 0

	f(x,y)

f(x,y) = log(x+y)
"""
f(x) = x<2 ? 0 : (x-2)^(1/2)

# ╔═╡ bda9f415-97a6-4350-a52b-942349cc6a6c
f(x,y) = log(x+y);

# ╔═╡ 8d45d70d-c934-4f91-8401-6d0db41e2164
"""
	gen_2d_coef(n)

Generate the polynomial basis coefficients for the 2-d function, where `n` is the number of points used in the equally-spaced range of numbers to sample from the true function.
"""
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

# ╔═╡ 00d1e352-1278-4aa1-8c54-ea0571c21520
"""
	gen_1d_coef(n)

Generate the polynomial basis coefficients for the 1-d function, where `n` is the number of points used in the equally-spaced range of numbers to sample from the true function.
"""
function gen_1d_coef(n)
    START=0.1; STOP=4.0
    X = range(START,STOP,n)
    Z = [f(x) for x ∈ X]
    b = PolyGetCoef(X,Z)
    return b
end

# ╔═╡ a9a80795-23d1-4b5c-8f3f-88d7736652a7
"""
	gen_2d_extrapolation(n, b; START=0.1, STOP=2.5)

inputs:

	n = number of points used
	b = vector of polynomial basis coefficients
	START = starting number for X and Y arrays
	STOP = stopping number for X and Y arrays

outputs:

	XX = array of n equally-spaced X points, repeated (inner) n times for plotting
	YY = array of n equally-spaced Y points, repeated (outer) n times for plotting
	Z = array of n function values = f(X)
	Zhat = array of n approximated function values using the basis coefficients b
	ε = array of n errors = Z - Zhat
	norm = maximum of the absolute value of the errors
"""
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

# ╔═╡ 369c2291-a68e-4fce-a974-023be7b18979
"""
	gen_1d_extrapolation(n, b; START=0.1, STOP=2.5)

Generate the X and Zhat arrays from the extrapolation over the array of `n` values in the domain of X (ranging from `START` to `STOP`), extrapolating using the previously fit polynomial basis represented by `b`.

inputs:

	n = number of points in the equally-spaced array of X points
	b = vector of polynomial basis coefficients
	START = starting number for X and Y arrays
	STOP = stopping number for X and Y arrays

outputs:

	X = array of n equally-spaced X points
	Z = array of n function values = f(X)
	Zhat = array of n approximated function values using the basis coefficients b
	ε = array of n errors = Z - Zhat
	norm = maximum of the absolute value of the errors
"""
function gen_1d_extrapolation(n, b; START=0.1, STOP=4.5)
    X = range(START,STOP,n)
    Z = [f(x) for x ∈ X]
    B = PolyBasis(X)
    Zhat = B*b
    ε = Z - Zhat
    norm = maximum(abs.(ε))
    return (X,), Z, Zhat, ε, norm
end

# ╔═╡ f3ad279b-fa20-4db9-8e7c-f3df7bba3494
"""
	plot_stuff(X, Z, Zhat, ε, n)

Plot the true function values, the approximated function values, and the error on the same plot, for the 1-d problem.
"""
function plot_stuff(X, Z, Zhat, ε, n)
    plot(X, Z, label="true value")
    plot!(X, Zhat, label="approx n=$n")
	plot!(X, zeros(size(X)), label="", color=:grey)
	plot!(X, ε, label="approx error",
		  title="True/Extrapolated and Error for f(x) [n=$n]")
end

# ╔═╡ 71650b4f-84c8-4e77-823e-d866d204f60b
begin
	# Generate the approximated values and error norms for the 2-d problem for n=5,15,35
	data_2d = Dict()
	for n ∈ [5, 15, 35]
		# Generate polynomial coefficients
		b = gen_2d_coef(n)
		# Calculate true, interpolated, and exterpolated function values
		_, _, _, _, norm_in = gen_2d_extrapolation(n, b; STOP=2.0)
		(X,Y), Z, Zhat, _, norm_out = gen_2d_extrapolation(n, b; STOP=2.5)
		data_2d[n] = Dict("X"=>X, "Y"=>Y, "Z"=>Z, "Zhat"=>Zhat,
			"norm in"=>norm_in, "norm out"=>norm_out)
		# println("n=$n, norm in = $norm_in, norm out = $norm_out")
	end
end

# ╔═╡ 150c3b98-9523-4182-9ae9-5d89ece56a5a
"""
	plot_surface(n)

Plot the true function values, the approximated function values, and the error on the same plot, for the 2-d problem for n=`n`.
"""
function plot_surface(n)
	surface(data_2d[35]["X"], data_2d[35]["Y"], data_2d[35]["Z"])
	s = surface!(data_2d[n]["X"], data_2d[n]["Y"], data_2d[n]["Zhat"],
		title="True and approx function values for f(x,y) [n=$n]",
		camera=(50,25),alpha = 0.5)
	return s
end

# ╔═╡ 692557e2-1048-4f68-8ca7-46fbc8e04597
md"""
# Approximation of a bivariate function
$$f(x,y) = log(x+y)$$
"""

# ╔═╡ 8c8c4521-e76c-4e4e-8d50-cc984e76a3bf
# ╠═╡ show_logs = false
plot_surface(5)

# ╔═╡ b7845383-8775-49b2-a172-34a797dec204
Markdown.parse("""
Max Interpolation Error = $(round(data_2d[5]["norm in"], digits=DIGITS))

Max Extrapolation Error = $(round(data_2d[5]["norm out"], digits=DIGITS))

---""")

# ╔═╡ f9cc37ae-e181-42f3-a4bf-dd3d16613e8b
# ╠═╡ show_logs = false
plot_surface(15)

# ╔═╡ 38ab0864-3c54-4fc9-bae7-363ad78be41c
Markdown.parse("""
Max Interpolation Error = $(round(data_2d[15]["norm in"], digits=DIGITS))

Max Extrapolation Error = $(round(data_2d[15]["norm out"], digits=DIGITS))

---""")

# ╔═╡ e0ed5a95-a84a-4280-8f99-cc2de6fa3576
# ╠═╡ show_logs = false
plot_surface(35)

# ╔═╡ f3a3587f-fece-4224-b1de-b4fe190f2d02
Markdown.parse("""
Max Interpolation Error = $(round(data_2d[35]["norm in"], digits=DIGITS))

Max Extrapolation Error = $(round(data_2d[35]["norm out"], digits=DIGITS))

---""")

# ╔═╡ 6f2086e7-fd20-4568-911f-7e72c3e4e4fa
md"""
# Approximation of a univariate function
$$f(x) = \begin{cases} 
0 & x<2 \\
(x-2)^{1/2} &x\geq 2\\
\end{cases}$$
"""

# ╔═╡ 5b752602-2f3c-4b30-a419-f224ddfb4072
begin
	# Plot n=5 true-function, approx-function, and error values
	# Generate polynomial coefficients and return true function values
	n=5
	b = gen_1d_coef(n)
	# Generate extrapolation
	X, Z, Zhat, ε, norm = gen_1d_extrapolation(n, b; STOP=5)
	# Plot true function, approximate, and error
	plot_stuff(X..., Z, Zhat, ε, n)
end

# ╔═╡ 3eb4f16f-c4cc-4439-ab14-bb08a67e1543
begin
	data_1d = Dict()
	for n ∈ [5, 15, 35]
		# Generate polynomial coefficients and return true function values
		b = gen_1d_coef(n)
		# Generate extrapolation
		_, _, _, _, norm_in = gen_1d_extrapolation(n, b; STOP=4.0)
		X, Z, Zhat, _, norm_out = gen_1d_extrapolation(n, b; STOP=5.0)
		data_1d[n] = Dict("X"=>X, "Z"=>Z, "Zhat"=>Zhat, "norm_in"=>norm_in, 
			"norm_out"=> norm_out)
	end
	# Get most detailed "true function values" from the last n=35 set of data
	p_1d = plot(data_1d[35]["X"], data_1d[35]["Z"], lw=2, color=:black,
		 label="true value",
		 title="Function Approximation for f(x) for different n",
		 legend=:bottomright,
		 xlabel="x", ylabel="f(x)")
	# Add the plots for the different levels of approximation
	for n ∈ [5, 15, 35]
		p_1d = plot!(data_1d[n]["X"], data_1d[n]["Zhat"], lw=2,
				  label="n=$n approximation")
	end
	p_1d
end

# ╔═╡ 727650ff-8a40-42c0-bd77-2cfee9b9edfc
md"""
## Max Abs Approximation Error
Extrapolation error: error calculated over extrapolated function values, outside of the domain of original coefficient fitting.

Interpolation error: error calculated over interpolated function values, inside of the domain of original coefficient fitting.
"""

# ╔═╡ 0046c5fe-0cfb-404a-a3ef-a78169632cef
begin
	norm_ins = [data_1d[n]["norm_in"] for n ∈ [5, 15, 35]]
	norm_outs = [data_1d[n]["norm_out"] for n ∈ [5, 15, 35]]
	DataFrame(Dict("n"=>[5, 15, 35],
		"Interpolation Max Abs Error"=>norm_ins,
		"Extrapolation Max Abs Error"=>norm_outs))
end

# ╔═╡ 9a90d955-d1ad-4c5b-b00e-774f1647270a
# ╠═╡ disabled = true
#=╠═╡
# begin
# 	# depreciated - used to print all plots for both 1d and 2d problems
# 	plots = []
# 	for n ∈ [5, 15, 35]
# 	    for dimensions in 1:2
# 	        println("n=$n, dim=$dimensions")
# 	        if dimensions == 1  # select 1 dimensional problem
# 	            gen_coef = gen_1d_coef
# 	            gen_extrap = gen_1d_extrapolation
# 	            stop = 5
# 	        elseif dimensions == 2  # select 2 dimensional problem
# 	            gen_coef = gen_1d_coef
# 	            gen_extrap = gen_1d_extrapolation
# 	            stop = 2.5
# 	        end
# 	        # Generate polynomial coefficients and return true function values
# 	        b = gen_coef(n)
# 	        # Generate extrapolation
# 	        X, Z, Zhat, ε, norm = gen_extrap(n, b; STOP=stop)
# 	        # Plot true function, approximate, and error
# 	        plot_stuff(X..., Z, Zhat, ε, n)
# 	    end
# 	end
# end
  ╠═╡ =#

# ╔═╡ f5a131a7-3f45-470a-87e0-099c810a7e55
# ╠═╡ disabled = true
#=╠═╡
# begin
# 	# depreciated - used to try to plot all 3d plots, but needed to plot separately
# 	data_2d = Dict()
# 	for n ∈ [5, 15, 35]
# 		# Generate polynomial coefficients and return true function values
# 		b = gen_2d_coef(n)
# 		# Generate extrapolation
# 		X, Z, Zhat, _, _ = gen_2d_extrapolation(n, b; STOP=stop)
# 		data_2d[n] = Dict("X"=>X, "Z"=>Z, "Zhat"=>Zhat)
# 	end
# 	# Get most detailed "true function values" from the last n=35 set of data
# 	p_2d = plot(data_2d[35]["X"], data_2d[35]["Z"], lw=2, color=:black,
# 		 label="true value",
# 		 title="Function Approximation for f(x)",
# 		 legend=:bottomright,
# 		 xlabel="x", ylabel="f(x)")
# 	# Add the plots for the different levels of approximation
# 	for n ∈ [5, 15, 35]
# 		p_2d = plot!(data_2d[n]["X"], data_2d[n]["Zhat"], lw=2,
# 				  label="n=$n approximation")
# 	end
# 	p_2d
# end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─b52f0010-5be8-11ed-04f2-63ec2d2c7f26
# ╟─50b76a1d-8cc2-4fa0-892a-ba19e345e5cf
# ╟─cbf11ac8-b1a9-4522-b09e-b926be6d382d
# ╠═bca3746e-7066-48a7-91cf-d149d0c04a45
# ╠═742ee234-a942-4a71-b739-e5a734392138
# ╟─76f05bd0-669b-4032-b2d2-5b24f1f1b821
# ╟─c207865e-dfcb-4312-9cd6-6a5719b1c206
# ╠═91a6674f-951b-4e4d-bcbc-429e50a508fb
# ╟─25f8282b-a45d-4d2e-8cef-90206baa3de1
# ╟─9f252393-3eb9-4749-8df8-381ab5c369bd
# ╟─bda9f415-97a6-4350-a52b-942349cc6a6c
# ╠═8d45d70d-c934-4f91-8401-6d0db41e2164
# ╠═00d1e352-1278-4aa1-8c54-ea0571c21520
# ╠═a9a80795-23d1-4b5c-8f3f-88d7736652a7
# ╠═369c2291-a68e-4fce-a974-023be7b18979
# ╟─f3ad279b-fa20-4db9-8e7c-f3df7bba3494
# ╠═150c3b98-9523-4182-9ae9-5d89ece56a5a
# ╠═71650b4f-84c8-4e77-823e-d866d204f60b
# ╟─692557e2-1048-4f68-8ca7-46fbc8e04597
# ╟─8c8c4521-e76c-4e4e-8d50-cc984e76a3bf
# ╟─b7845383-8775-49b2-a172-34a797dec204
# ╟─f9cc37ae-e181-42f3-a4bf-dd3d16613e8b
# ╟─38ab0864-3c54-4fc9-bae7-363ad78be41c
# ╟─e0ed5a95-a84a-4280-8f99-cc2de6fa3576
# ╟─f3a3587f-fece-4224-b1de-b4fe190f2d02
# ╟─6f2086e7-fd20-4568-911f-7e72c3e4e4fa
# ╠═5b752602-2f3c-4b30-a419-f224ddfb4072
# ╠═3eb4f16f-c4cc-4439-ab14-bb08a67e1543
# ╟─727650ff-8a40-42c0-bd77-2cfee9b9edfc
# ╟─0046c5fe-0cfb-404a-a3ef-a78169632cef
# ╠═9a90d955-d1ad-4c5b-b00e-774f1647270a
# ╠═f5a131a7-3f45-470a-87e0-099c810a7e55

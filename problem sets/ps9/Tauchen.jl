try
    using Distributions
catch e
    import Pkg; Pkg.add(["Distributions"])
    using Distributions
end
cdf_normal(x) = cdf(Normal(),x)

"""
Function tauchen(N,mu,rho,sigma,m)

    Purpose:    Finds a Markov chain whose sample paths
                approximate those of the AR(1) process
                    z(t+1) = (1-rho)*mu + rho * z(t) + eps(t+1)
                where eps are normal with stddev sigma

    Format:     {Z, Zprob} = Tauchen(N,mu,rho,sigma,m)

    Input:      N       scalar, number of nodes for Z
                mu      scalar, unconditional mean of process
                rho     scalar
                sigma   scalar, std. dev. of epsilons
                m       max +- std. devs.

    Output:     Z       N*1 vector, nodes for Z
                Zprob   N*N matrix, transition probabilities

        Martin Floden
        Fall 1996

        This procedure is an implementation of George Tauchen's algorithm
        described in Ec. Letters 20 (1986) 177-181.
"""
function tauchen(N,mu,rho,sigma,m)
    Zprob = zeros(N,N);
    a     = (1-rho)*mu;

    ZN = m * sqrt(sigma^2 / (1 - rho^2))
    Z = range(-ZN + mu, ZN + mu, N)
	zstep = Z[2]-Z[1]
    
    for j ∈ 1:N, k ∈ 1:N
		if k == 1
			Zprob[j,k] = cdf_normal((Z[1] - a - rho * Z[j] + zstep / 2) / sigma)
		elseif k == N
			Zprob[j,k] = 1 - cdf_normal((Z[N] - a - rho * Z[j] - zstep / 2) / sigma)
		else
			Zprob[j,k] = cdf_normal((Z[k] - a - rho * Z[j] + zstep / 2) / sigma) - 
						 cdf_normal((Z[k] - a - rho * Z[j] - zstep / 2) / sigma);
		end
    end
    
    return Z, Zprob
end






##############################################
#               unused functions
##############################################


"""
Function tauchen2(N, μ, ρ, σ, m)

    Purpose:    Finds a Markov chain whose sample paths
                approximate those of the AR(1) process
                    log(Y(t+1)) = (1-ρ)*log(μ) + ρ * log(Y(t)) + ε(t+1)
     want to match       Z(t+1) = (1-ρ)*mu     + ρ * Z(t)      + ε(t+1)

                where ε are normal with stddev σ
                so: mu = log(μ) and Z = log(Y)

    Format:     {Y, Yprob} = (N, μ, ρ, σ, m)

    Input:      N       scalar, number of nodes for Z
                μ       scalar, unconditional mean of Y
                ρ       scalar
                σ       scalar, std. dev. of epsilons
                m       max +- std. devs.

    Output:     Y       N*1 vector, equally-spaced range/nodes for Y
                Yprob   N*N matrix, transition probabilities

        Martin Floden
        Fall 1996

        This procedure is an implementation of George Tauchen's algorithm
        described in Ec. Letters 20 (1986) 177-181.
"""
function tauchen2(N, μ, ρ, σₑ, m)
    mu = log(μ)
    Yprob = zeros(N,N);
    a     = (1-ρ)*mu;

    # Calculate endpoints of Z = log(Y) grid
    σz = (σₑ / (1-ρ^2))^(1/2)
    Zᴺ = m*σz; Z¹ = -m*σz
    # Convert endpoints to Y = e^Z
    Yᴺ = exp(Zᴺ); Y¹ = exp(Z¹)
    # Create regular grid in Y
    Y = range(Y¹, Yᴺ, N)
    # Calculate midpoints of Y grid
    ystep = Y[2]-Y[1]
    ymidpoints = Y[1:(end-1)] + step/2
    # Convert grid to Z = log(Y)
    Z = log.(Y)
    # Calculate probabilities for Z grid (same as probabilities for Y)
    # Return regular grid of Y and Y probabilities

	w = Z[2]-Z[1]
    
    for j ∈ 1:N, k ∈ 1:N
		if k == 1
			Zprob[j,k] = cdf_normal((Z[1] + w/2 - a - ρ*Z[j]) / σₑ)
		elseif k == N
			Zprob[j,k] = 1 - cdf_normal((Z[N] - w/2 - a - ρ*Z[j]) / σₑ)
		else
			Zprob[j,k] = cdf_normal((Z[k] + w/2 - a - ρ*Z[j]) / σₑ) - 
						 cdf_normal((Z[k] - w/2 - a - ρ*Z[j]) / σₑ);
		end
    end
    
    return Z, Zprob
end



























"""Return probability of moving from Y-state j to Y-state k."""
function Yprob_jk(j, k, Y, F, wup, wdown)
    halfup = 
    if 1 < k < length(Y)
        return F(Y[k] - ρ*Y[j] + wup) - F(Y[k] - ρ*Y[j] - wdown)
    elseif k == 1
        return F(Y[k] - ρ*Y[j] + wup)
    elseif k == length(Y)
        return 1 - F(Y[k] - ρ*Y[j] - wdown)
    else
        error("j ($j) or k ($k) not valid!")
    end
end


"""
Generate discrete values and markov transition matrix for Y where
    y[t+1] = (1-ρ)*μ + ρ * y[t] + ε[t+1]
    actual equation of motion: 
        log(y[t+1]) = (1-ρ)*μ + ρ * log(y[t]) + ε[t+1]
        Y' = μ^(1-ρ) * Y^ρ * ℯ^ε

    N  = number of discrete states of Y
    μ  = unconditional mean of Y
    ρ  = the AR(1) decay factor
    σₑ = the std dev of ε noise terms
    m  = # of std dev's of Y to cover by the descrete values

    Y  = N*1 vector, nodes for Y
    YP = N*N matrix, transition probabilities, columns sum to 1
         the (k,j) element is the probability of moving from j to k.
"""
function mytauchen(N, μ, ρ, σₑ, m)
    # define the CDF of ε
    Fₑ(u) = cdf(Normal(0, σₑ),x)
    # Generate the discrete values for Z = Y-μ (mean 0)
    σy = (σₑ^2 / (1-ρ^2))^(1/2)
    Yᴺ = m*σy
    Y¹ = -Yᴺ
    Z = range(Y¹, Yᴺ, N)
    w = Z[2]-Z[1]
    # Generate markov transition matrix for Y
    YP = [Yprob_jk(j, k, Z, F, w) for j ∈ 1:N, k ∈ 1:N]
    # Shift Y to be centered on the unconditional mean μ
    Y = range(Y¹ + μ, Yᴺ + μ, N)
    return Y, YP
end

"""
Generate discrete values and markov transition matrix for Y where
    log(y[t+1]) = (1-ρ)*log(μ) + ρ * log(y[t]) + ε[t+1]
    actual equation of motion: 
        log(y[t+1]) = (1-ρ)*μ + ρ * log(y[t]) + ε[t+1]
        Y' = μ^(1-ρ) * Y^ρ * ℯ^ε

    N  = number of discrete states of Y
    μ  = unconditional mean of Y
    ρ  = the AR(1) decay factor
    σₑ = the std dev of ε noise terms
    m  = # of std dev's of Y to cover by the descrete values

    Y  = N*1 vector, nodes for Y
    YP = N*N matrix, transition probabilities, columns sum to 1
         the (k,j) element is the probability of moving from j to k.
"""
function mytauchen(N, μ, ρ, σₑ, m)
    # define the CDF of ε
    Fₑ(u) = cdf(Normal(0, σₑ),u)
    # Calculate the end points of log(Y)
    σlny = (σₑ^2 / (1-ρ^2))^(1/2)
    lnYᴺ = m*σlny
    lnY¹ = -lnYᴺ
    # Convert to endpoints for Y
    Yᴺ = exp(lnYᴺ); Y¹ = exp(lnY¹)
    # Generate the equally-spaced discrete values for Z = Y-μ (mean 0)
    Z = range(Y¹, Yᴺ, N)
    w = Z[2]-Z[1]
    # Convert range of Z to range of 


    # Generate markov transition matrix for Y
    YP = [Yprob_jk(j, k, Z, F, w) for j ∈ 1:N, k ∈ 1:N]
    # Shift Y to be centered on the unconditional mean μ
    Y = range(Y¹ + μ, Yᴺ + μ, N)
    return Y, YP
end





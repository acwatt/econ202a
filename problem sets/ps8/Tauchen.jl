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
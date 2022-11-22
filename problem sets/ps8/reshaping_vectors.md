I realized that I just needed to reverse the order of building the meshgrid from
```julia
AY = [(a,y) for a ∈ GridA for y ∈ GridlnY]
```
to 
```julia
AY = [(a,y) for y ∈ GridlnY for a ∈ GridA]
```
so when we use `reshape(V, nA, nY)` later, it correctly assigns the elements. We could also use `reshape(V, nY, nA)'` to transpose the matrix and get the same elements, but we also need to transpose the EV matrix before unfurlling: `EV'[:]`, which is just more confusing for the reader.


**OLD:**


I think that the reshaping of the V vector to calculate EV might be incorrect -- at least, if you created the meshgrid vectors like I did. My meshgrid vectors look like below, where its iterating over:
```julia
AY = [(a,y) for a ∈ GridA for y ∈ GridlnY]
AA = [a for (a,y) ∈ AY]
YY = [y for (a,y) ∈ AY]

julia> [AA YY]
700×2 Matrix{Float64}:
  2.50333  -0.458831
  2.50333  -0.305888
  2.50333  -0.152944
  2.50333   0.0
  2.50333   0.152944
  2.50333   0.305888
  2.50333   0.458831
  3.46421  -0.458831
  3.46421  -0.305888
  ⋮
 96.669     0.458831
 97.6299   -0.458831
 97.6299   -0.305888
 97.6299   -0.152944
 97.6299    0.0
 97.6299    0.152944
 97.6299    0.305888
 97.6299    0.458831
```

The goal: reshape V into a 100x7 matrix to multiply by the 7x7 markov transition matrix for Y. This means we want the 100x7 V matrix to be rows
```julia
a
```


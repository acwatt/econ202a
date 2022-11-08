function [Astruct] = buildA_(borrowingLimit, amax, ajump, incomeWorkMin, incomeWorkMax)
%Create Savings Account A
%    Let A_ denote the vector of all possible A_t values in the simulation
%    This section builds variable A_.


		%----------
        %NOTE: Lower ajump means a better approximation of continuous 
		%	   savings acct,but comes at cost of higher run-time
        %      Similarly, higher xmax means agent allowed to save more, 
        %      but also comes at cost of higher run-time
		%----------
		
    A_ = [borrowingLimit:ajump:amax];
    alen = length(A_);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %A few necessary variables / conditions
	%	Note: if these conditons do not hold, it likely stems from the borrowingLimit
	%		We discretize X, starting at borrowingLimit and ending at xmax. The conditions
	%		below ensure that we hit certain points in that discretized state space
    assert( ~isempty( find(A_ == 0) ) )
	assert( ~isempty( find(A_ == incomeWorkMin) ) )
	assert( ~isempty( find(A_ == incomeWorkMax) ) )
    azeroIndex = find(A_ == 0);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	Astruct = struct('A_', A_, ...
					'alen', alen, ... 
					'azeroIndex', azeroIndex, ...
					'ajump', ajump,  ...
					'amax', amax, ...
					'borrowingLimit', borrowingLimit);
	
end


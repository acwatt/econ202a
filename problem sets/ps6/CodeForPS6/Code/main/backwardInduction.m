function [EV__,  Ix__, V__, W__, C__, Edelta_c__, Evar_delta_c__] = ...
	backwardInduction(lifecycleStruct, returnStruct, Astruct, incomeStruct);

%{
##########################################################################
##########################################################################
UPDATE 1/8/17: Now allowing for sophisticated and naive quasi-hyperbolic discounting


##########################################################################
##########################################################################
%backwardInduction.m

----------------------
NOTE: I admit the syntax of this code may be difficult to follow. Standard for-loops in 
	Matlab are too slow, so this code is partially vectorized to speed run-time
----------------------


	
The logic of this file is as follows --

Starting in the final year of life:
    Calculate utility for all possible values of X
        --> C_T = A_T in final year of life

Looping Backward from year t = EOL - 1 to t = 1:    
    1. Calculate self t's value function for all values of A, as follows:
            
			For each value of A the agent may have in period t, denoted a_t:
				For each possible savings value from borrowingLimit to xmax:
                    a. Calculate instantaneous utility u( a_t - savings)
                    b. Calculate current value function from instantaneous
                    utility and expected continuation payoff
                END
				Calculate the optimum behaviour over all savings values for agent with assets a_t
			END
	--> For all possible values a_t in period t, we have now calculated optimal savings / consumption behaviour
			
    2. Calculate expected continuation payoff for agents in period t-1:
        a. We already knew agent t+1's value function for each asset level a_t+1
		b. In step 1 we calculated policy and value functions for each asset level a_t
		c. As long as we know the income process (we do), we have all we need to calculate 
			the expected continuation payoff for agents in period t-1 who pass assets 
			of R*(a_{t-1} - c_{t-1}) to period t

    3. Repeat loop for year t-1
        
NOTE: In the code below I do not actually write the for-loop described in step 1 above. 
It's too slow, so I vectorize the for-loop for efficiency.


At the end of this file, we also calculate:
	1. The expected change in log-consumption from period t to t+1
	2. The expected variance of this change in log-consumption

##########################################################################
##########################################################################
%}

%%-------------------------------------------------------------------------
	%%Unpack the structs
	yearsWork = lifecycleStruct.yearsWork;
	yearsAlive = lifecycleStruct.yearsAlive;
	eolRepay = lifecycleStruct.eolRepay;
	
	delta = returnStruct.delta;
	beta = returnStruct.beta;
	beta_hat = returnStruct.beta_hat;
	R = returnStruct.R;
	util = returnStruct.util;
	
	A_ =  Astruct.A_;	
	ajump = Astruct.ajump;
    alen = Astruct.alen;
	azeroIndex =  Astruct.azeroIndex;
	
	incomeCase = incomeStruct.incomeCase;
	incomeWorkMin = incomeStruct.incomeWorkMin;
	incomeWorkMax = incomeStruct.incomeWorkMax;
	incomeWorkRange = incomeStruct.incomeWorkRange;
%%-------------------------------------------------------------------------


%%-------------------------------------------------------------------------
%{ 
Define variables to store results of backward induction, as follows:
    NOTE: all matrices defined in this section have xlen rows and
    yearsAlive columns. Thus, they store the behaviour calculated at every
    possible set of state variables (A, t)

    1. EV__ stores the expected continuation payoff.
		IMPORTANT: the indexing of EV__ is different than these other variables.
			EV__ is indexed based on (a_t-c_t), not a_t.
			This makes sense, because EV__ is the continuation payoff, which is 
				based on the amount of assets passed to the next period
    2. Ix__ stores the index (from 1 to xlen) of optimal savings
    3. C__ stores consumption
    4. V__ stores the continuation value function
	5. W__ stores the current value function
		-->The difference between V and W is only relevant 
			when agent is a quasi-hyperbolic discounter
%}

    EV__ = zeros(alen, yearsAlive);         
    Ix__ = zeros(alen, yearsAlive);
    V__  = zeros(alen, yearsAlive);
	W__  = zeros(alen, yearsAlive);
    C__  = zeros(alen, yearsAlive);

        %Pre-allocate space to save time
        payoffs__ = zeros(alen, alen);
%%-------------------------------------------------------------------------

    


%%-------------------------------------------------------------------------
%RUN BACKWARD INDUCTION
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Vectorize the calculation of instantaneous utility to save runtime
        %{
		----------------------
		Because the calculation of instantaneous utility from all possible values of
			consumption is independent of t, we calculate outside of the for-loop
			to speed runtime
		----------------------
		
        possibleSavings__ is a 2-dimensional matrix, with the columns
            representing all possible savings values to ensure that in
            period t+1, agent has assets in the discretized set [borrowingLimit:xjump:xmax]
 
        A__ is a 2-dimensional matrix, with the rows representing all
            possible asset values an agent can have in period t
    
        possibleConsumption__ combines A__ and possibleSavings__ to
            calculate the consumption of an agent who enters with assets
            defined by A__ and saves the values set by possibleSavings__
            -->possibleConsumption__ set to 0 whenever savings >= consumption
        
        instantaneousUtility__ stores the instantaneous utility from all
            consumption values in possibleConsumption__
        %}
    possibleSavings__ = repmat(A_ ./ R, alen, 1);
    A__ = repmat(A_', 1, alen);
    possibleConsumption__ = A__ - possibleSavings__;
    possibleConsumption__(possibleConsumption__<0) = 0;
    instantaneousUtility__ = util(possibleConsumption__);
    instantaneousUtility__(instantaneousUtility__ <= util(0)) = -Inf;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Loop over state variables t to calculate optimal decision in each state (t, X)
    for t = yearsAlive:-1:1
        t
		
        %- - - - - - - - - - - - - - - - 
        %Define EOL value function
        if t == yearsAlive
			if eolRepay == 1 
				%Agent not allowed to die with negative assets
				%	How do we ensure this? Set a value of -infinity for doing so
				if azeroIndex > 1
					EV__(1:(azeroIndex-1), end) = -Inf ;
				end
			else
				%do nothing, just leave EV__(:,end) = 0 for all x;
			end	
        end
        %- - - - - - - - - - - - - - - - 
        
        %- - - - - - - - - - - - - - - - 
        %Calculate optimal behaviour for each value of X in year t
        
		
		%Each column in possibleConsumption__ represents a different value of (x_t - c_t),
		%	as defined by variable possibleSavings__. Since EV__ is indexed by values of 
		%	(x_t - c_t), we can simply add EV_(:,t)' to instantaneousUtility__.
		%	This is what is done below.
		expectedContPayoff_ = delta * EV__(:,t)';
		payoffs__ = bsxfun(@plus, instantaneousUtility__,  beta * expectedContPayoff_);
			%Slower (but easier to read) version of bsxfun:
				%expectedContPayoff__ = repmat(expectedContPayoff_, xlen, 1);
				%payoffs__ = instantaneousUtility__ + expectedContPayoff__;
		
        %Each row represents a different possible value of x_t, so
        %calculate the max payoff from each row
		%W__ is current value function; V__ is continuation value function
        [optPayoff_, optIx_] = max(payoffs__, [], 2);
        W__(:, t) = optPayoff_;
		if (beta ~= 1)
			%If quasi-hyperbolic discounting, back out continuation value 
			% 	from current value by removing beta-discounting on t+1 payoff
			instUtilityIndex_ = sub2ind(size(instantaneousUtility__), [1:alen], optIx_');
			instantaneousUtilityRealized_ = instantaneousUtility__(instUtilityIndex_)';
				expectedContPayoffRealized_ = (optPayoff_ - instantaneousUtilityRealized_) ./ beta;
			V__(:,t) = instantaneousUtilityRealized_ + expectedContPayoffRealized_;
			V__(find(isnan(V__))) = -Inf;
		else
			%If exponential discounting, no difference between continuation value function
			%	and current value function
			V__ = W__;
		end
		Ix__(:,t) = optIx_;
        consumptionIndex_ = sub2ind(size(possibleConsumption__), [1:alen], optIx_');
        C__(:,t) = possibleConsumption__(consumptionIndex_);
            C__(find(optPayoff_ == -Inf), t) = -Inf;
			
		%Naive Discounting
		%	Continuation value function based on beta-hat, the expected beta-discount factor
		%	of all future selves
		if (beta_hat ~= beta)
			payoffs_hat__ = bsxfun(@plus, instantaneousUtility__,  beta_hat * expectedContPayoff_);
			[optPayoff_hat_, optIx_hat_] = max(payoffs_hat__, [], 2);
				instUtilityIndex_hat_ = sub2ind(size(instantaneousUtility__), [1:alen], optIx_hat_');
				instantaneousUtilityRealized_hat_ = instantaneousUtility__(instUtilityIndex_hat_)';
				expectedContPayoffRealized_hat_ = (optPayoff_hat_ - instantaneousUtilityRealized_hat_) ./ beta_hat;
			V__(:,t) = instantaneousUtilityRealized_hat_ + expectedContPayoffRealized_hat_;
			V__(find(isnan(V__))) = -Inf;
		end
        %- - - - - - - - - - - - - - - - 
        
        %- - - - - - - - - - - - - - - -
        %Calculate expected continuation payoff for t-1 agents
        if t > 1
            %%Generate EV__ for period t-1 based on period t decisions
            switch incomeCase
                case 1
                    %Income is deterministic, so there is no uncertainty
                    if t > yearsWork
                        %Income = 0 in current period t
                        EV__(:,t-1) = V__(:,t);
                    elseif t <= yearsWork
                        %Income = 1 in current period t
                        %   -->Thus an agent who saves 0 enters period t+1
                        %   with X = 1. We index V__ accordingly
                        incomeSteps = incomeWorkMax / ajump;
                        EV__(:,t-1) = [V__(incomeSteps+1:end,t); (V__(end, t).*ones(incomeSteps, 1))];
                    end

                case 2
                    if t > yearsWork
                        %Income = 0 in current period t
                        EV__(:,t-1) = V__(:,t);
                    elseif t <= yearsWork
                        %Income no longer deterministic, so, for each X, we
                        %must average over all possible period t income values
                        minIncomeSteps = incomeWorkMin / ajump;
                        possibleIncomeValues = incomeWorkRange / ajump;
                        for ix = 1:alen
                            startInd = min(ix + minIncomeSteps, alen);
                            endInd = startInd+possibleIncomeValues;
							if endInd <= alen
								EV__(ix, t-1) = sum(V__(startInd:endInd,t)) / (endInd - startInd + 1);
							else
								%Winsorize assets + income at xmax
								EV__(ix, t-1) = (sum(V__(startInd:alen,t)) + (endInd - alen) * V__(alen, t)) / (endInd - startInd + 1);
							end
                        end    
                    end
            end
        end
        %- - - - - - - - - - - - - - - - 
    end
%%-------------------------------------------------------------------------




%%-------------------------------------------------------------------------
% In order to check our model's simulation against Euler equation conditions, 
%	calculate the following for all possible state variables (x,t):
% 		1. The expected change in log-consumption from period t to t+1
%		2. The expected variance of this change in log-consumption
[Edelta_c__, Evar_delta_c__] = expectedConsumption_tplus1(C__, lifecycleStruct, returnStruct, Astruct, incomeStruct);
%%-------------------------------------------------------------------------
end

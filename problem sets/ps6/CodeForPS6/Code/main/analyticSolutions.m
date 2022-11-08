function [V__, C__] = analyticSolutions(lifecycleStruct, returnStruct, Astruct, incomeStruct)

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
    amax = Astruct.amax;
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

    1. C__ stores consumption
    2. V__ stores the continuation value function
	
%}

    V__  = zeros(alen, yearsAlive);
    C__  = zeros(alen, yearsAlive);
    
    
    for a = 1:alen
        for t = 1:yearsAlive
            
        %Consumption Policy Function
        C__(a,t) = incomeWorkMax + A_(a)*(1+R)*(1-delta)/(1-delta^(yearsAlive-t+1));
        
        if C__(a,t)< 0
           C__(a,t) = -Inf;
        end
        
        %Value Function
        if C__(a,t)< 0
            V__(a,t) = -Inf;
        else
            V__(a,t) = (1- delta^(yearsAlive-t+1))/(1-delta)*util(C__(a,t));
        end
        
        end
    end
    
    %%Clean up
    C__(C__<0) = -Inf;
    
end
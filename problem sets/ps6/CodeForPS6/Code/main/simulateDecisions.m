function [simY__,  realY__, simA__, realA__, simSavings__, realSavings__, realC__] = simulateDecisions(C__, Ix__, pop, lifecycleStruct, returnStruct, Astruct, incomeStruct)

%{
##########################################################################
##########################################################################
%simulateDecisions.m

This file simulates consumer behavior based on the policy functions that
were created in the script backwardInduction.m


    Note on variable naming convention: the prefix 'real' stands for
    realized, and marks *actual values*. The prefix 'sim' stands for
    simulated, and marks the *index of values*. The 'real' variables are
    useful for tracking what is going on in the simulation, but all of the
    work is done by the 'sim' variables. 


##########################################################################
##########################################################################
%}

	%%-------------------------------------------------------------------------
	%%Unpack the structs
		yearsWork = lifecycleStruct.yearsWork;
		yearsRet = lifecycleStruct.yearsRet;
		yearsAlive = lifecycleStruct.yearsAlive;
				
		ajump = Astruct.ajump;
		alen = Astruct.alen;
		azeroIndex =  Astruct.azeroIndex;
		
		incomeCase = incomeStruct.incomeCase;
		incomeWorkMin = incomeStruct.incomeWorkMin;
		incomeWorkMax = incomeStruct.incomeWorkMax;
		incomeWorkRange = incomeStruct.incomeWorkRange;

		
		
	%%-------------------------------------------------------------------------
		%Simulate Income Process for all agents

		switch incomeCase
			case 1
				realY__ = [incomeWorkMax * ones(pop, yearsWork), zeros(pop, yearsRet)];
				simY__ = realY__ ./ ajump;
			case 2
				%Note the need to round income to values of xjump 
				numIncomeValues = 1 + (incomeWorkRange ./ajump);
				randYWork__ = floor(numIncomeValues * rand(pop, yearsWork));
				realY__ = [incomeWorkMin + (ajump .* randYWork__), zeros(pop, yearsRet)];
				simY__ = round(realY__ ./ ajump);     
		end
	%%-------------------------------------------------------------------------

	%%-------------------------------------------------------------------------
	%{ 
	Define variables to store the behavior each individual agent
		--> each row is an agent, each column is a year

		1. realC__ stores each agent's realized consumption values
		2. simX__ stores the *index* of X_t, where X_t is the level of assets
				that an agent enters period t with
		3. simSavings__ stores the *index* of each agent's saving amount in
				period t
	%}

		realC__ = zeros(pop, yearsAlive);
		simA__ = zeros(pop, yearsAlive);
		simSavings__ = zeros(pop, yearsAlive);


	%{
	%Simulate agent behavior based on the following dynamic budget constraint:
		-->X_t+1 = R(X_t -c_t) + y_t+1
	%}

		%Seed model -- all agents enter year 1 with only an income realization
		simA__(:,1) = azeroIndex+ simY__(:,1);
		
		%Given X_t, calculate X_t+1 for each agent
		for t = 1:yearsAlive
			%Calculate savings and consumption in period t
			simSavings__(:, t) = Ix__(simA__(:, t), t);
			realC__(:, t) = C__(simA__(:,t), t);
			
			%Given savings and consumption, calculate index of X_t+1
			if t < yearsAlive
				simA__(:,t+1) = min(simSavings__(:,t) + simY__(:, t+1), alen);
			end
		end    
		
		%Convert sim into real for output
		realA__ = (simA__ - azeroIndex) .* ajump;
		realSavings__ = ((simA__ - azeroIndex) - simY__) .* ajump;
	%%-------------------------------------------------------------------------
end	
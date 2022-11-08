%{
##########################################################################
##########################################################################
Originally written by: Peter Maxted
Updated: Ethan M. L. McClure

Code for Simple Buffer Stock Model

Model Outline:
Let Y denote the income earned by the agent.
        -->  Deterministic y = 1 during work;

Let A_t represent liquid assets in period t (after earning income Y)
        -->A_t has a minimum (credit limit) set by variable borrowingLimit
        -->A_t earns a real rate of return R

Let I_t denote investment into the liquid asset X during period t

Dynamic Budget Constraint: A_t+1 = R(A_t - C_t) + Y
Borrowing Constraint:  A_t+1 - Y = R(A_t - C_t) >= credit limit

------------------------------------------------------------------
State Variables: A_t, t

Choice Variable: I_t
        -->Consumption is calculated as a residual
------------------------------------------------------------------


Agent Decision Framework:
Agent's have log utility and discount at rate delta

Let V(A_t+1) denote the continuation payoff of self t (given state variables A_t+1, t+1)
	--> Self t's Bellman equation is: max[ u(C_t) + delta * E(V(A_t+1)) ]

UPDATE 1/8/17: Now allowing for sophisticated and naive quasi-hyperbolic discounting.
	--> If specified, self t's Bellman equation is: max[ u(C_t) + beta-hat* delta * E(V(A_t+1)) ]


Simulation:
The code first generates policy and value functions at every combination of state variables.
	--> See function backwardInduction and code backwardInduction_general.m 
	--> E.g., variables C__, V__

Using these policy and value functions, profiles for 5000 simulated agents are generated 
	--> See function simulateDecisions and code simulateDecisions.m
	--> E.g., variables realC__, realY__

##########################################################################
##########################################################################
%}

clear all
addpath('main')
addpath('check') %Includes a small code that checks cases 1-3 with analytic solutions
addpath('graphs_for_lecture') %Links to a code that generates graphs used in lecture

%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
%{
%%Pre-specified cases (Steps toward a simple buffer stock model)
    1. [NS] Deterministic y = 1 during work and y = 0 during infinite
    retirement. R = 1.05. delta*R = 1. Log utility. T = 51;

    2. [NS] Deterministic y = 1 during work and y = 0 during infinite
    retirement. R = 1.05. delta*R = 1. Log utility. T = 251;
    
    3. [NS] Deterministic y = 1 during work and y = 0 during infinite
    retirement. R = 1.05. delta*R = 1. Isoelstic utility, rho = .5.    

    4. [NS] Deterministic y = 1 during work and y = 0 during infinite
    retirement. R = 1.01. delta*R = 1. Log utility. T = 51;
%}

step = 1;
switch step
    case 1
        yearsWork = 51;
        yearsRet = 0;
        incomeCase = 1;
        R = 1.05;       % Interest Rate on X
        delta = 1/R;		
		%borrowingLimit = (set below to not bind);
        amax = 50;              %set so irrelevant
        plotType = 1;
        ies = 1;
        
    case 2
        yearsWork = 251;
        yearsRet = 0;
        incomeCase = 1;
        R = 1.05;       % Interest Rate on X
        delta = 1/R;			%R = 1.05 (defined below)
		%borrowingLimit = (set below to not bind);
        amax = 50;              %set so irrelevant
        plotType = 1;  
        ies = 1;

    case 3
        yearsWork = 51;
        yearsRet = 0;
        incomeCase = 1;
        R = 1.05;       % Interest Rate on X
        delta = 1/R;			%R = 1.05 (defined below)
		%borrowingLimit = (set below to not bind);
        amax = 50;              %set so irrelevant
        plotType = 1;
        ies = .5;
        
    case 4
        yearsWork = 51;
        yearsRet = 0;
        incomeCase = 1;
        R = 1.01;       % Interest Rate on X
        delta = 1/R;			%R = 1.05 (defined below)
		%borrowingLimit = (set below to not bind);
        amax = 50;              %set so irrelevant
        plotType = 1;
        ies = 1;

end


%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
%% DEFINE MODEL PARAMETERS
%
%		Note: Code below uses structs to group variables. Not necessary for this code,
%		but will hopefully make extensions to this code slightly easier
%
%		Note: Commented-out variables below are defined as part of the above cases.
%		Included for conceptual clarity
%

    %- - - - - - - - - - - - - - - - 
    %Define Model Framework
    pop = 1;                     %Size of simulated population

    %yearsWork = ;                 % # years agent works
    %yearsRet = ;                  % # years agent lives during retirement
	yearsAlive = yearsWork + yearsRet;
	
	%As a default, set eolRepay = 1  (eol = End Of Life)
	%	eolRepay = 1 is No-Ponzi Condition. Agents must die with non-negative assets
	if exist('eolRepay', 'var') ~= 1
		eolRepay = 1;
	else
		eolRepay %Already defined
	end
	
	lifecycleStruct = struct('yearsWork', yearsWork, 'yearsRet', yearsRet, 'yearsAlive', yearsAlive, 'eolRepay', eolRepay);
    %- - - - - - - - - - - - - - - - 
    
    %- - - - - - - - - - - - - - - - 
    %Define Case Specific Income Process
    %    1. Deterministic y = 1 during work; y = 0 during retirement
    %    2. E[y] = 1 during work with f(y); y = 0 during retirement
    %incomeCase = ;
	incomeStruct = buildIncome(incomeCase);
		incomeWorkMin = incomeStruct.incomeWorkMin;
		incomeWorkMax = incomeStruct.incomeWorkMax;
		incomeWorkRange = incomeStruct.incomeWorkRange;
    %- - - - - - - - - - - - - - - - 
    
    %- - - - - - - - - - - - - - - - 
    %Euler Equation Parameters
    %delta = ;                  	% Discount Factor
	if exist('beta', 'var') ~= 1
		%If beta not defined, default to exponential discounting
		beta = 1;
	else
		beta; %Already defined
	end
	if exist('beta_hat', 'var') ~= 1
		%If beta-hat (naivete) not defined, default to sophisticated
		beta_hat = beta;
	else
		beta_hat; %Already defined
    end
	
    if ies == 1
        util = @(x) log(x);             % Utility Fxn -- Log utility
    else
        util = @(x) (x.^(1-ies) - 1)./(1-ies);% Utility Fxn -- Isoelastic utility
    end
	
	returnStruct = struct('R', R, 'delta', delta, 'beta', beta, 'beta_hat', beta_hat, 'util', util);
    %- - - - - - - - - - - - - - - - 
    
    %- - - - - - - - - - - - - - - - 
    %Savings Account X Parameters
	
	%As a default, set the borrowing limit so that it doesn't bind
	%	i.e., set the borrowing limit low enough that agents will never choose to hit it
	%	How can we do this? Set borrowingLimit = -1 * (incomeWorkMin / (R-1)) - incomWorkMin.
	%	Why? If the agent earns incomWorkMin in period t+1, then his income is only enough to 
	%		pay interest on debt, leaving the principal unchanged. 
	%		If the agent lives for a finite number of years, there is a non-zero probability she
	%		will earn incomeWorkMin in ALL remaining periods, and therefore must consume a non-positive
	%		amount in order to repay debts. However, the Inada condition prevents the agent from
	%		ever choosing this path. Hence, the borrowing limit does not bind.
	
	if exist('borrowingLimit', 'var') ~= 1
		borrowingLimit = round(-1 * (incomeWorkMin / (R-1)) - incomeWorkMin, 6);
	else
		borrowingLimit %Already defined
	end
	
    %xmax = ;                      % Maximum in Savings Acct
    
	ajump = 0.025;                   % xjump discretizes the state space
		if step >= 5
			ajump = .005;
        end
    
	Astruct = buildA_(borrowingLimit, amax, ajump, incomeWorkMin, incomeWorkMax);
		A_ =  Astruct.A_;
		alen = Astruct.alen;
		azeroIndex =  Astruct.azeroIndex;	
    %- - - - - - - - - - - - - - - - 
%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
%%%Analytic Solution

    %Call script to run backward induction
	[V__analytic, C__analytic] = analyticSolutions(lifecycleStruct, returnStruct, Astruct, incomeStruct);


%%-------------------------------------------------------------------------
%%%BACKWARD INDUCTION

    %Call script to run backward induction
	[EV__,  Ix__, V__, W__, C__, Edelta_c__, Evar_delta_c__] = ...
		backwardInduction(lifecycleStruct, returnStruct, Astruct, incomeStruct);

    %Call script to simulate agent decisions given the policy functions
    %that were built in backwardInduction.m
	[simY__,  realY__, simA__, realA__, simSavings__, realSavings__, realC__] = ...
		simulateDecisions(C__, Ix__, pop, lifecycleStruct, returnStruct, Astruct, incomeStruct);
%%-------------------------------------------------------------------------


%%-------------------------------------------------------------------------
%%Plot Consumer Behavior
%%		Below are some sample plots of consumer behavior, for reference.
%%		These plots should be updated by the user as desired
    
    %plotType = ;
    
    %- - - - - - - - - - - - - - - -
    switch plotType
        %% Finite Horizon Plots
        case 1
            
            subplot(3,2,1)
            %Numerical Value Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, V__(:, 1:yearsWork));
            xlabel('Age')
            ylabel('A_t')
            zlabel('V_t')
            title('Numerical Value Function')
            set(h,'LineStyle','none')
			colormap jet
            
            subplot(3,2,2)
            %Numerical Consumption Policy Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, C__(:, 1:yearsWork));
            xlabel('Age')
            ylabel('A_t')
            zlabel('C_t')
            title('Numerical Consumption Policy Function')
            set(h,'LineStyle','none')
			colormap jet
            
            subplot(3,2,3)
            %Analytic Value Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, V__analytic(:, 1:yearsWork));
            xlabel('Age')
            ylabel('A_t')
            zlabel('V_t')
            title('Analytic Value Function')
            set(h,'LineStyle','none')
			colormap jet
            
            subplot(3,2,4)
            %Analytic Consumption Policy Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, C__analytic(:, 1:yearsWork));
            xlabel('Age')
            ylabel('A_t')
            zlabel('C_t')
            title('Analytic Consumption Policy Function')
            set(h,'LineStyle','none')
			colormap jet
            
            subplot(3,2,5)
            %Numerical Value Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, abs(V__(:, 1:yearsWork) - V__analytic(:, 1:yearsWork)));
            xlabel('Age')
            ylabel('A_t')
            zlabel('abs(errors)')
            title('Value Function Errors')
            set(h,'LineStyle','none')
			colormap jet
            
            subplot(3,2,6)
            %Numerical Consumption Policy Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, abs(C__(:, 1:yearsWork) - C__analytic(:, 1:yearsWork)));
            xlabel('Age')
            ylabel('A_t')
            zlabel('abs(errors)')
            title('Consumption Policy Function Errors')
            set(h,'LineStyle','none')
			colormap jet
     
     
		otherwise
		%do nothing
    end
    %- - - - - - - - - - - - - - - -
%%-------------------------------------------------------------------------



%%-------------------------------------------------------------------------
%%Check that borrowingLimit/xmax were set large enough
if (max(max(realA__)) >= 0.85 * amax)
    display('ERROR -- Agents potentially bounded by X account maximum. Re-run with a higher xmax')
end
if (min(min(realSavings__)) <= 0.85 * borrowingLimit & borrowingLimit ~= round(-1 * (incomeWorkMin / (R-1)) - incomeWorkMin, 6) & step ~= 6)
    display('ERROR -- Agents potentially bounded by X account minimum. Re-run with a lower borrowing limit unless constraint desired')
end
%%-------------------------------------------------------------------------



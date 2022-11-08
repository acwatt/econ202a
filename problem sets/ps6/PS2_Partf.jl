"""
File: PS2_Partf.jl
Author: AC Watt
Date: 2022-10-27

# Purpose
Recreation of PS2_Partf.m matlab code originally written 
by Peter Maxted and updated by Ethan M. L. McClure in 2017

Changes by Aaron
- simplified code by removing all parts that were not changing by only changing STEP
- moved all remaining functions into this script (from the extra functions in the main folder)

# Description (from PS2_Partf.m)
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
"""

#============================================================================
                DIFFERENCES BETWEEN JULIA AND MATLAB
============================================================================
Structs:
- Matlab uses structs - those exist in Julia but you can't add struct fields on the fly like in matlab
- Instead we'll use dictionaries, which will operate nearly the same is structs in the matlab code

Switch - case:
- this is the same as if, elseif, else.
=#

#============================================================================
                                IMPORTS
============================================================================#
using Random



#============================================================================
                            SET GLOBAL PARAMETERS
============================================================================#
# Set global variables determining run. See set_globals for description of STEP
# global variables set: yearsWork, yearsRet, yearsAlive, incomeCase, R, delta, amax, plotType, ies, pop, eolRepay
STEP = 1  # ∈{1,2,3,4}

#= Pre-specified cases (Steps toward a simple buffer stock model)
    STEP = 1: [NS] Deterministic y = 1 during work and y = 0 during infinite
    retirement. R = 1.05. delta*R = 1. Log utility. T = 51;

    STEP = 2: [NS] Deterministic y = 1 during work and y = 0 during infinite
    retirement. R = 1.05. delta*R = 1. Log utility. T = 251;
    
    STEP = 3: [NS] Deterministic y = 1 during work and y = 0 during infinite
    retirement. R = 1.05. delta*R = 1. Isoelstic utility, rho = .5.    

    STEP = 4: [NS] Deterministic y = 1 during work and y = 0 during infinite
    retirement. R = 1.01. delta*R = 1. Log utility. T = 51;
=#
println("\nSetting globals based on STEP = $STEP")
if STEP == 1
    yearsWork = 51;   # # years agent works
    R = 1.05;         # Interest Rate on X
    ies = 1;
elseif STEP == 2
    yearsWork = 251;  # # years agent works
    R = 1.05;         # Interest Rate on X
    ies = 1;
elseif STEP == 3
    yearsWork = 51;   # # years agent works
    R = 1.05;         # Interest Rate on X
    ies = .5;
elseif STEP == 4
    yearsWork = 51;   # # years agent works
    R = 1.01;         # Interest Rate on X
    ies = 1;
else
    throw(DomainError(STEP, "STEP must be an integer 1-4"))
end

# Set constant parameters
pop = 1;          # Size of simulated population
yearsRet = 0;     # # years agent lives during retirement
delta = 1/R;      # Discount Factor
amax = 50;        # set so irrelevant
plotType = 1;
yearsAlive = yearsWork + yearsRet;
# As a default, set eolRepay = 1  (eol = End Of Life)
# 	eolRepay = 1 is No-Ponzi Condition. Agents must die with non-negative assets
eolRepay = 1

# Deterministic y = 1 during work; y = 0 during retirement
incomeWorkMax = 1
incomeWorkMin = 1
incomeWorkRange = incomeWorkMax - incomeWorkMin

lifecycleStruct = Dict("yearsWork" => yearsWork, "yearsRet" => yearsRet, "yearsAlive" => yearsAlive, "eolRepay" => eolRepay)



#- - - - - - - - - - - - - - - - 
#Euler Equation Parameters
# default to exponential discounting
beta = 1;
# default to sophisticated
beta_hat = beta;

# Utility function
if ies == 1
    util(x) = log(x)             # Log utility
else
    util(x) = (x.^(1-ies) - 1)./(1-ies); # Isoelastic utility
end

# returnStruct = struct('R', R, 'delta', delta, 'beta', beta, 'beta_hat', beta_hat, 'util', util);
    
#- - - - - - - - - - - - - - - - 
#Savings Account X Parameters

#As a default, set the borrowing limit so that it doesn't bind
#	i.e., set the borrowing limit low enough that agents will never choose to hit it
#	How can we do this? Set borrowingLimit = -1 * (incomeWorkMin / (R-1)) - incomeWorkMin.
#	Why? If the agent earns incomeWorkMin in period t+1, then his income is only enough to 
#		pay interest on debt, leaving the principal unchanged. 
#		If the agent lives for a finite number of years, there is a non-zero probability she
#		will earn incomeWorkMin in ALL remaining periods, and therefore must consume a non-positive
#		amount in order to repay debts. However, the Inada condition prevents the agent from
#		ever choosing this path. Hence, the borrowing limit does not bind.

borrowingLimit = -1 * (incomeWorkMin / (R-1)) - incomeWorkMin
	
#xmax = ;                      % Maximum in Savings Acct

ajump = 0.025;                   # xjump discretizes the state space

#Create Savings Account A
#    Let A_ denote the vector of all possible A_t values in the simulation
#    This section builds variable A_.


		#----------
        #NOTE: Lower ajump means a better approximation of continuous 
		#	   savings acct,but comes at cost of higher run-time
        #      Similarly, higher xmax means agent allowed to save more, 
        #      but also comes at cost of higher run-time
		#----------
		
    A_ = borrowingLimit:ajump:amax
    alen = length(A_)
    azeroIndex = find(A_ == 0)
    
	Astruct = buildA_(borrowingLimit, amax, ajump, incomeWorkMin, incomeWorkMax);
		A_ =  Astruct.A_;
		alen = Astruct.alen;
		azeroIndex =  Astruct.azeroIndex;	
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Analytic Solution

    %Call script to run backward induction
	[V__analytic, C__analytic] = analyticSolutions(lifecycleStruct, returnStruct, Astruct, incomeStruct);


#-------------------------------------------------------------------------
#BACKWARD INDUCTION

    %Call script to run backward induction
	[EV__,  Ix__, V__, W__, C__, Edelta_c__, Evar_delta_c__] = ...
		backwardInduction(lifecycleStruct, returnStruct, Astruct, incomeStruct);

    %Call script to simulate agent decisions given the policy functions
    %that were built in backwardInduction.m
	[simY__,  realY__, simA__, realA__, simSavings__, realSavings__, realC__] = ...
		simulateDecisions(C__, Ix__, pop, lifecycleStruct, returnStruct, Astruct, incomeStruct);
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
#Plot Consumer Behavior
#		Below are some sample plots of consumer behavior, for reference.
#		These plots should be updated by the user as desired
    
    #plotType = ;
    
    #- - - - - - - - - - - - - - - -
    switch plotType
        # Finite Horizon Plots
        case 1
            
            subplot(3,2,1)
            #Numerical Value Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, V__(:, 1:yearsWork));
            xlabel("Age")
            ylabel("A_t")
            zlabel("V_t")
            title("Numerical Value Function")
            set(h,"LineStyle","none")
			colormap jet
            
            subplot(3,2,2)
            #Numerical Consumption Policy Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, C__(:, 1:yearsWork));
            xlabel("Age")
            ylabel("A_t")
            zlabel("C_t")
            title("Numerical Consumption Policy Function")
            set(h,"LineStyle","none")
			colormap jet
            
            subplot(3,2,3)
            #Analytic Value Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, V__analytic(:, 1:yearsWork));
            xlabel("Age")
            ylabel("A_t")
            zlabel("V_t")
            title("Analytic Value Function")
            set(h,"LineStyle","none")
			colormap jet
            
            subplot(3,2,4)
            #Analytic Consumption Policy Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, C__analytic(:, 1:yearsWork));
            xlabel("Age")
            ylabel("A_t")
            zlabel("C_t")
            title("Analytic Consumption Policy Function")
            set(h,"LineStyle","none")
			colormap jet
            
            subplot(3,2,5)
            #Numerical Value Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, abs(V__(:, 1:yearsWork) - V__analytic(:, 1:yearsWork)));
            xlabel("Age")
            ylabel("A_t")
            zlabel("abs(errors)")
            title("Value Function Errors")
            set(h,"LineStyle","none")
			colormap jet
            
            subplot(3,2,6)
            #Numerical Consumption Policy Function
            age_ = [20:20+yearsAlive-1];
            h = surf(age_(1:yearsWork), A_, abs(C__(:, 1:yearsWork) - C__analytic(:, 1:yearsWork)));
            xlabel("Age")
            ylabel("A_t")
            zlabel("abs(errors)")
            title("Consumption Policy Function Errors")
            set(h,"LineStyle","none")
			colormap jet
     
     
		otherwise
		#do nothing
    end
    #- - - - - - - - - - - - - - - -
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
#Check that borrowingLimit/xmax were set large enough
if (max(max(realA__)) >= 0.85 * amax)
    display('ERROR -- Agents potentially bounded by X account maximum. Re-run with a higher xmax')
end
if (min(min(realSavings__)) <= 0.85 * borrowingLimit & borrowingLimit ~= round(-1 * (incomeWorkMin / (R-1)) - incomeWorkMin, 6) & step ~= 6)
    display('ERROR -- Agents potentially bounded by X account minimum. Re-run with a lower borrowing limit unless constraint desired')
end
#-------------------------------------------------------------------------








#============================================================================
                                FUNCTIONS
============================================================================#




function [incomeStruct] = buildIncome(incomeCase)
    %Define Case-Specific Income Process
    %    1. Deterministic y = 1 during work; y = 0 during retirement
    %    2. E[y] = 1 during work with f(y); y = 0 during retirement
    
        switch incomeCase
            
            case 1
                incomeWorkMax = 1
                incomeWorkMin = 1
                incomeWorkRange = incomeWorkMax - incomeWorkMin;
                    assert(incomeWorkRange == 0)
    
            case 2                       
                incomeWorkMax = 1.5
                incomeWorkMin = 0.5
                incomeWorkRange = incomeWorkMax - incomeWorkMin;
                rng(1864) %set seed
        end
        
        incomeStruct = struct('incomeCase', incomeCase, ...
                            'incomeWorkMin', incomeWorkMin, ...
                            'incomeWorkMax', incomeWorkMax, ...
                            'incomeWorkRange', incomeWorkRange);
        
    
    end
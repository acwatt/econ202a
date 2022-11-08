function [Edelta_c__, Evar_delta_c__] = expectedConsumption_tplus1(C__, lifecycleStruct, returnStruct, Astruct, incomeStruct)

%%-------------------------------------------------------------------------
% In order to check our model's simulation against Euler equation conditions, 
%	calculate the following for all possible state variables (x,t):
% 		1. The expected change in log-consumption from period t to t+1
%		2. The expected variance of this change in log-consumption
%%-------------------------------------------------------------------------

%%-------------------------------------------------------------------------
%%Unpack the structs
	yearsWork = lifecycleStruct.yearsWork;
	yearsAlive = lifecycleStruct.yearsAlive;
	
	R = returnStruct.R;
	
	A_ =  Astruct.A_';
	ajump = Astruct.ajump;
	alen = Astruct.alen;
	azeroIndex =  Astruct.azeroIndex;
	
	incomeCase = incomeStruct.incomeCase;
	incomeWorkMin = incomeStruct.incomeWorkMin;
	incomeWorkMax = incomeStruct.incomeWorkMax;
	incomeWorkRange = incomeStruct.incomeWorkRange;


%Pre-allocate space to save time
	Edelta_c__ = zeros(alen, yearsAlive-1);
	Evar_delta_c__ = zeros(alen, yearsAlive-1);
		
%Calculate logs outside of for-loop to save time
	c_t__ = real(log(C__));		
%%-------------------------------------------------------------------------

%%-------------------------------------------------------------------------		
%1. Expected change in log-consumption from period t to t+1
	for t = 1:yearsAlive-1
	 switch incomeCase
		case 1
			%Income is deterministic, so there is no uncertainty
			if t >= yearsWork
				%Income = 0 in period t+1
				c_t = c_t__(:,t);
				savings = R*(A_ - C__(:, t));
                idx = (savings < Inf) & ([0;diff(savings)] <0);
                validSavingsStartIndex = min(find(idx)); 
				validSavings = savings(validSavingsStartIndex:end);
					savings_index = round((validSavings ./ ajump) + azeroIndex);		
					%Agent earns 0 income in next period
					next_pd_asset_index = min(savings_index + 0, alen);							
					c_tplus1 = c_t__(next_pd_asset_index, t+1);
					Edelta_c__(validSavingsStartIndex:end, t) = c_tplus1 - c_t(validSavingsStartIndex:end);
				
					Edelta_c__(1:validSavingsStartIndex-1, t) = NaN;
				
			elseif t < yearsWork
				%Income = 1 in period t+1
				incomeSteps = incomeWorkMax / ajump;
				
				c_t = c_t__(:,t);
				savings = R*(A_ - C__(:, t));
                %min(find(savings < Inf));
                idx = (savings < Inf) & ([0;diff(savings)] <0);
                validSavingsStartIndex = min(find(idx)); 
				validSavings = savings(validSavingsStartIndex:end);
					savings_index = round((validSavings ./ ajump) + azeroIndex);
					next_pd_asset_index = min(savings_index + incomeSteps, alen);
					c_tplus1 = c_t__(next_pd_asset_index, t+1);
					Edelta_c__(validSavingsStartIndex:end, t) = c_tplus1 - c_t(validSavingsStartIndex:end);
				
					Edelta_c__(1:validSavingsStartIndex-1, t) = NaN;
			end

		case 2
			if t >= yearsWork
				%Income = 0 in period t+1
				c_t = c_t__(:,t);
				savings = R*(A_ - C__(:, t));
				validSavings = savings(find(savings < Inf));
				validSavingsStartIndex = min(find(savings < Inf));
					savings_index = round((validSavings ./ ajump) + azeroIndex);		
					%Agent earns 0 income in next period
					next_pd_asset_index = min(savings_index + 0, alen);							
					c_tplus1 = c_t__(next_pd_asset_index, t+1);
					Edelta_c__(validSavingsStartIndex:end, t) = c_tplus1 - c_t(validSavingsStartIndex:end);
				
					Edelta_c__(1:validSavingsStartIndex-1, t) = NaN;
				
			elseif t < yearsWork
				%Income no longer deterministic. For each X, we
				%must average over all possible period t income values
				minIncomeSteps = incomeWorkMin / ajump;
				possibleIncomeValues = incomeWorkRange / ajump;
				scaling = 1 / (possibleIncomeValues + 1);
						
				c_t = c_t__(:,t);
				savings = R*(A_ - C__(:, t));
				validSavings = savings(find(savings < Inf));
				validSavingsStartIndex = min(find(savings < Inf));
				savings_index = round((validSavings ./ ajump) + azeroIndex);
				
				%%----------
				%(Vectorized code below)
				%Average consumption over all possible income states in t+1
				iy_ = [0:possibleIncomeValues] + minIncomeSteps;
				next_pd_asset_index = bsxfun(@plus, savings_index, iy_);
				next_pd_asset_index = min(next_pd_asset_index, alen);
				possible_c_tplus1__ = c_t__(next_pd_asset_index', t+1);
					%Currently a column vector, so re-shape back to 2d array
					%	-->Matlab reshapes column-wise and we want row-wise, so following code is a little strange
					possible_c_tplus1__ = reshape(possible_c_tplus1__, size(next_pd_asset_index,2), size(next_pd_asset_index,1));
					possible_c_tplus1__ = possible_c_tplus1__';
				Ec_tplus1_ = scaling * sum(possible_c_tplus1__, 2);
					
					%The (slower) for-loop for above block of code
					%Ec_tplus1 = zeros(length(savings_index), 1);
					%for iy = 0:possibleIncomeValues
					%	next_pd_asset_index = min(savings_index + minIncomeSteps + iy, xlen);
					%	Ec_tplus1 = Ec_tplus1 + scaling * log(C__(next_pd_asset_index, t+1));
					%end
				%%----------
				
				Edelta_c__(validSavingsStartIndex:end, t) = Ec_tplus1_ - c_t(validSavingsStartIndex:end);
				Edelta_c__(1:validSavingsStartIndex-1, t) = NaN;
			end
		end
	end		
%%-------------------------------------------------------------------------




%%-------------------------------------------------------------------------		
%2. Expected variance of change in log-consumption		
	for t = 1:yearsAlive-1
	 switch incomeCase
		case 1
			%Income is deterministic, so there is no uncertainty
			%Only question is whether consumption is defined
			savings = R*(A_ - C__(:, t));
			validSavings = savings(find(savings < Inf));
			validSavingsStartIndex = min(find(savings < Inf));
			Evar_delta_c__(validSavingsStartIndex:end, t) = 0;	
			
			Evar_delta_c__(1:validSavingsStartIndex-1, t) = NaN;


		case 2
			if t >= yearsWork
				%Income = 0 (deterministic)in t+1, so there is no uncertainty
				savings = R*(A_ - C__(:, t));
				validSavings = savings(find(savings < Inf));
				validSavingsStartIndex = min(find(savings < Inf));
				Evar_delta_c__(validSavingsStartIndex:end, t) = 0;
			
				Evar_delta_c__(1:validSavingsStartIndex-1, t) = NaN;
				
			elseif t < yearsWork
				%Income no longer deterministic
				minIncomeSteps = incomeWorkMin / ajump;
				possibleIncomeValues = incomeWorkRange / ajump;
				scaling = 1 / (possibleIncomeValues + 1);
						
				c_t = c_t__(:,t);
				savings = R*(A_ - C__(:, t));
				validSavings = savings(find(savings < Inf));
				validSavingsStartIndex = min(find(savings < Inf));
				savings_index = round((validSavings ./ ajump) + azeroIndex);
				
				%%----------
				%(Vectorized code below)
				%Calculate Delta log-consumption over all possible income states in t+1
				iy_ = [0:possibleIncomeValues] + minIncomeSteps;
				next_pd_asset_index = bsxfun(@plus, savings_index, iy_);
				next_pd_asset_index = min(next_pd_asset_index, alen);
				possible_c_tplus1__ = c_t__(next_pd_asset_index', t+1);
					%Currently a column vector, so re-shape back to 2d array
					%	-->Matlab reshapes column-wise and we want row-wise, so following code is a little strange
					possible_c_tplus1__ = reshape(possible_c_tplus1__, size(next_pd_asset_index,2), size(next_pd_asset_index,1));
					possible_c_tplus1__ = possible_c_tplus1__';
				delta_c_tplus1__ = bsxfun(@minus, possible_c_tplus1__, c_t(validSavingsStartIndex:end));
				
				Evar_numerator__ = (bsxfun(@minus, delta_c_tplus1__, Edelta_c__(validSavingsStartIndex:end,t))).^2;
				%%----------
				
				Evar_delta_c__(validSavingsStartIndex:end, t) = scaling .* sum(Evar_numerator__, 2);
				Evar_delta_c__(1:validSavingsStartIndex-1, t) = NaN;
			end
		end
	end
end		
		
		
		
		
		
		
		
%%-------------------------------------------------------------------------
%%OLD (NON-VECTORIZED) CODE FOR REFERENCE	
%{
%1. Expected change in log-consumption from period t to t+1
for t = 1:yearsAlive-1
	for ix = 1:xlen
		 switch incomeCase
                case 1
                    %Income is deterministic, so there is no uncertainty
                    if t >= yearsWork
                        %Income = 0 in period t+1
                        c_t = log(C__(ix, t));
						savings = R*(X__(ix, t) - C__(ix, t));
						if savings ~= Inf
							savings_index = round((savings ./ xjump) + xzeroIndex);		
							%Agent earns 0 income in next period
							next_pd_asset_index = min(savings_index + 0, xlen);							
							c_tplus1 = log(C__(next_pd_asset_index, t+1));
							Edelta_c(ix, t) = c_tplus1 - c_t;
						else
							Edelta_c(ix, t) = NaN;
						end
                    elseif t < yearsWork
                        %Income = 1 in period t+1
                        incomeSteps = incomeWorkMax / xjump;
						
						c_t = log(C__(ix, t));
						savings = R*(X__(ix, t) - C__(ix, t));
						if savings ~= Inf
							savings_index = round((savings ./ xjump) + xzeroIndex);
							next_pd_asset_index = min(savings_index + incomeSteps, xlen);
							c_tplus1 = log(C__(next_pd_asset_index, t+1));
							Edelta_c(ix, t) = c_tplus1 - c_t;
						else
							Edelta_c(ix, t) = NaN;
						end
                    end

                case 2
                    if t >= yearsWork
                        %Income = 0 in period t+1
                        c_t = log(C__(ix, t));
						savings = R*(X__(ix, t) - C__(ix, t));
						if savings ~= Inf
							savings_index = round((savings ./ xjump) + xzeroIndex);		
							%Agent earns 0 income in next period
							next_pd_asset_index = min(savings_index + 0, xlen);							
							c_tplus1 = log(C__(next_pd_asset_index, t+1));
							Edelta_c(ix, t) = c_tplus1 - c_t;
						else
							Edelta_c(ix, t) = NaN;
						end
                    elseif t < yearsWork
                        %Income no longer deterministic. For each X, we
                        %must average over all possible period t income values
                        minIncomeSteps = incomeWorkMin / xjump;
                        possibleIncomeValues = incomeWorkRange / xjump;
						scaling = 1 / (possibleIncomeValues + 1);
                        
						c_t = log(C__(ix, t));
						savings = R*(X__(ix, t) - C__(ix, t));
						if savings ~= Inf
							savings_index = round((savings ./ xjump) + xzeroIndex);
							Ec_tplus1 = 0;
							for iy = 0:possibleIncomeValues
								next_pd_asset_index = min(savings_index + minIncomeSteps + iy, xlen);
								Ec_tplus1 = Ec_tplus1 + scaling * log(C__(next_pd_asset_index, t+1));
							end
							Edelta_c(ix, t) = Ec_tplus1 - c_t;
						else
							Edelta_c(ix, t) = NaN;
						end  
                    end
            end
	end
end
%}
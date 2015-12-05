% Cited: Copyright (c) 2013, Chao Qu, Gareth Cross, Pimkhuan Hannanta-Anan. All
% rights reserved.
function [H] = calc_entropy(p)
% CALC_ENTROPY - Calculates the entropy of a probability distribution p.
%
%   p should be a 1 x N vector of probability vlues, where N is the no. 
%   of observations.
%

p_times_logp = min(0, p .* log2(p));
H = -sum(p_times_logp);

end

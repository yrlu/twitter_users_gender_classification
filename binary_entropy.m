function [H] = binary_entropy(p)
% BINARY_ENTROPY - Compute H(P(X)) for binary X.
%
% Usage:
% 
%    H = binary_entropy(P)
%
%  Returns the entroy H = -(P * log(P)  (1-P)*log(1-P)). If P is an M x N
%  matrix, H is also M x N, where computed element-wise for each P(i,j).

% Note: we need to correct for Matlab's insistence that 0 * -Inf = NaN.
% For entropy computation, 0 * log(0) = 0. This trick works because
% min(0,NaN) = 0, and log(P) <= 0 when P <= 1.
p_times_logp = @(x) min(0, x.*log2(x));

H = - (p_times_logp(p) + p_times_logp(1-p));
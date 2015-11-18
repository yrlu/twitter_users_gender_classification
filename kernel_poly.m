function K = kernel_poly(X, X2, p)
% Evaluates the Polynomial Kernel with specified degree p 
%
% Usage:
%
%    K = KERNEL_POLY(X, X2, P)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the polynomial kernel
% with degree P.

% HINT: This should be a very straightforward one liner!
K = ((X*(X2')).^p)';

% After you've computed K, make sure not a sparse matrix anymore
K = full(K);

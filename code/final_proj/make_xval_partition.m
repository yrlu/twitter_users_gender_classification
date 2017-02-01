function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

permutations = randperm(n);
part = mod(permutations, n_folds)+1;
part = part';
%     part = zeros(n, 1);
%     
%     fold = 0;
%     
%     eachfold = int16(n/n_folds);
%     
%     for i = 1 : n_folds
%          count = 0;
%          if i == n_folds
%              eachfold = n - int16(eachfold*(n_folds-1));
%          end
%          while count < eachfold
%              r = randi([1,n],1,1);
%              if part(r) == 0 
%                 part(r) = i;
%                 count = count + 1;
%              end
%          end
%     end
    
    
    
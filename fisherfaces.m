function [coeff, score,mu] = fisherfaces(X, Y)
% fisherfaces(X, Y, num_compo) returns the component sidentified by LDA and
% the projection of X (scores) 
%
%
%Args:
  %    X [num_data x dim] input data
  %    y [num_data x 1] classes
  %    pca_coeff, pca_score precalculated pca coefficient and scores
  %  
  %  Returns:
  %       coeff [] components identified by LDA
  %       scores [] apply coeff to X data
% 
% Modified by D.W Nov, 28
% see ref: https://github.com/bytefish/facerec/blob/master/m/models/fisherfaces.m

N = size(X,1); %num_data
C = 2; %number of classes 
mu = mean(X);

% num_comp = c-1; % at most 1

% reduce dim(X) to (N-c) (see paper [BHK1997])
%  Pca = pca(X, (N-c));
%  Lda = lda(project(X, Pca.W, Pca.mu), y, num_components);
  [pca_comp, pca_score, ~] = pca(X,'NumComponents',(N-C));
  % -> pca_comp = pca_coeff(:,1:(N-C)); could be precalculated 
  % -> pca_score = pca_scores(:,1:(N-C));
  % caution, pass transposed value to lda  
  lda_comp = lda(pca_score', Y');
  
  % build model
  coeff = pca_comp*lda_comp;
  score = X*coeff;
end
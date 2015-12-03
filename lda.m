function lda_w = lda(X, Y)
  %  Performs a Linear Discriminant Analysis and returns the 
  %  num_components components sorted descending by their 
  %  eigenvalue. 
  %
  %  num_components is bound to the number of classes, hence
  %  num_components = min(c-1, num_components)
  %  *In this specific 2-class example, num_components = 1;
  %
  %  Args:
  %    X [dim x num_data] input data
  %    y [1 x num_data] classes
  %    num_components [int] number of components to keep
  %  
  %  Returns:
  %    model [struct] Represents the learned model.
  %      .name [char] name of the model
  %      .num_components [int] number of components in this model
  %      .W [array] components identified by LDA
  %
  % Modified by D.W,  Nov,28
  % ref: https://github.com/bytefish/facerec/blob/master/m/models/lda.m
  dim = size(X,1);
  c = 2; 
  
  num_components = c-1;
  
  meanTotal = mean(X,2);
  
  Sw = zeros(dim, dim);
  Sb = zeros(dim, dim);
  for i=0:c-1
    Xi = X(:,Y==i);
    meanClass = mean(Xi,2);
    % center data
    Xi = Xi - repmat(meanClass, 1, size(Xi,2));
    % calculate within-class scatter
    Sw = Sw + Xi*Xi';
    % calculate between-class scatter
    Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
  end

  % solve the eigenvalue problem
  [V, D] = eig(Sb,Sw);
  
  % sort eigenvectors descending by eigenvalue
  [D,idx] = sort(diag(D), 1, 'descend');
  
  V = V(:,idx);
  % build model
  % model.D = D(1:num_components);
  lda_w = V(:,1:num_components);
end
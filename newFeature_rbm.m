function [ new_feat,new_feat_test ] = newFeature_rbm( dbn, train_x, test_x )

    
    % unfold dbn to nn
    nn = dbnunfoldtonn(dbn, size(train_x,2));
    nn.activation_function = 'sigm';
%     nn.learningRate = 5;
%     nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
%     nn.scaling_learningRate = 0.9;
%     
    % Get new training data features
    nn2 = nnff(nn, train_x, train_x);
    new_feat=nn2.a{2};
    
    % Get new test data features
    nn3 = nnff(nn, test_x, test_x);
    new_feat_test = nn3.a{2};

end


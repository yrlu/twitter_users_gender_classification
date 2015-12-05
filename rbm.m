function [ dbn ] = rbm( train_x )
    
    %  ex2 train a DBN. Its weights can be used to initialize a NN.
    rand('state',0)
    %train dbn
    dbn.sizes = [1000, 2;
    opts.numepochs =  10;
    opts.batchsize = 100;
%     opts.weightPenaltyL2 = 1e-2;  %  L2 weight decay
%     opts.scaling_learningRate = 1;
    opts.momentum  =   0;
    opts.alpha     =   1;
    dbn = dbnsetup(dbn, train_x, opts);
    dbn = dbntrain(dbn, train_x, opts);
    % figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

end


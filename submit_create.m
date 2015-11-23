new_features = [words_train,image_features_train];
IG=calc_information_gain(genders_train,new_features, [1:size(new_features,2)],10);
[top_igs, idx]=sort(IG,'descend');

words_train_s=new_features;% words_train;
words_test_s = [words_test, image_features_test];

cols_sel=idx(1:1000);
    
Xtrain=words_train_s(:,cols_sel);
Ytrain=genders_train;
    
Xtest=words_test_s(:,cols_sel);
    
ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree' ); %ens=regularize(ens);
    %ens = fitensemble(Xtrain,Ytrain, 'RobustBoost',300,'Tree','RobustErrorGoal',0.01,'RobustMaxMargin',1);
    %
Yhat= predict(ens,Xtest);
dlmwrite('submit.txt',Yhat, '\n');
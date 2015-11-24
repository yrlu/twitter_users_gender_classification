[n, ~] = size(words_train);
[parts] = make_xval_partition(n, 4);

%bns = calc_bns(words_train,Y);
IG=calc_information_gain(genders_train,words_train,[1:5000],10);
[top_bans, idx]=sort(IG,'descend');
%words_train_s=bsxfun(@times,words_train,IG);
acc = zeros(4,20);
for i=1:4
    disp('Testing fold', i, '..\n');
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:500);
    
    Xtrain=words_train(row_sel1,cols_sel);
    Ytrain=genders_train(row_sel1);
    Xtest=words_train(row_sel2,cols_sel);
    Ytest=genders_train(row_sel2);
    
    % test 20-200 clusters
    for j = 1:10
        disp('Testing cluster #', i*20, '..\n');
        Yhat = gaussian_mixture(Xtrain,Ytrain,Xtest,Ytest, j*20);
        confusionmat(Ytest,Yhat)
        acc(i,j)=sum(round(Yhat)==Ytest)/length(Ytest);
    end
end
acc;
mean(acc);
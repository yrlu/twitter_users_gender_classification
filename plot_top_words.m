% Deng, Xiang
%11/26
clear all
close all
load .\data\words_train.mat
load .\data\words_test.mat
load .\data\genders_train.mat
%% Plot Top words


X = [words_train ]; 
Y = genders_train;
bns = calc_bns(X,Y,0.01);
IG=calc_information_gain(Y,X,[1:size(X,2)],10);
[top_igs, idx_ig]=sort(IG,'descend');
[top_bns, idx_bns]=sort(bns,'descend');



[vid, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

words_counts=sum(X);
top_voc_ig=voc(idx_ig(1:1000));
top_voc_bns=voc(idx_bns(1:1000));



Nt=30;
figure
ig_sel=idx_ig(1:Nt);
bns_sel=idx_bns(1:Nt);
cnt_top_ig=words_counts(ig_sel);
cnt_top_bns=words_counts(bns_sel);
subplot(1,2,1);
plot(top_igs(1:Nt),cnt_top_ig,'ro')
text(top_igs(1:Nt),cnt_top_ig,voc(ig_sel),'Fontsize',15)
xlabel('IG')
ylabel('word occurences');
title ('Top words, IG')
subplot(1,2,2);
plot(top_bns(1:Nt),cnt_top_bns,'bo')
text(top_bns(1:Nt),cnt_top_bns,voc(bns_sel),'Fontsize',15)

xlabel(' BNS')
ylabel('word occurences');
title ('Top words,  BNS')
%%legend('IG','BNS')
%% Compared: IG and BNS
% Cited: Copyright (c) 2013, Chao Qu, Gareth Cross, Pimkhuan Hannanta-Anan. All
% rights reserved.
% number of points
N = 1000;

ig_vals = linspace(min(IG),max(IG),N);
bns_vals = linspace(min(bns),max(bns),N);
counts_ig = zeros(N,1);
counts_bns = zeros(N,1);

for i=1:N
   
    count_ig = nnz(IG >= ig_vals(i));
    count_bns = nnz(bns >= bns_vals(i));
    
    counts_ig(i) = count_ig;
    counts_bns(i) = count_bns;
    
end

figure;
subplot(2, 1, 1);
semilogy(ig_vals / max(ig_vals), counts_ig, 'b');
hold on;
semilogy(bns_vals / max(bns_vals), counts_bns, 'r');
title('Compared: IG and BNS');
ylabel('# of features retained');
xlabel('Threshold (Normalized)');
legend('IG', 'BNS');

subplot(2, 1, 2);
loglog(counts_ig, ig_vals, 'b');
hold on;
loglog(counts_bns, bns_vals, 'r');
title('Compared: IG and BNS');
xlabel('# of features retained');
ylabel('Threshold (Non-normalized)');
legend('IG', 'BNS');




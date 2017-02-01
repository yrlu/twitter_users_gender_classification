%% Plots/submission for SVM portion, Question 1.
addpath ./libsvm
%% Put your written answers here.
clear all
answers{1} = 'Answer to 1.3: The intersection kernel performs best. The similarity measurement is based on the frequency of the common words appears in both documents; it tells us how common the words appears in both documents, in the context of newsgroup post classification, it is reasonable that similar topics will have more common words that show up frequently and unrelevant topics will have limited common words occurs with high frequency. But this is just slightly better than the linear classifier';

save('problem_1_answers.mat', 'answers');

%% Load and process the data.

load ../data/windows_vs_mac.mat;
[X Y] = make_sparse(traindata, vocab);
[Xtest Ytest] = make_sparse(testdata, vocab);

%% Bar Plot - comparing error rates of different kernels

% INSTRUCTIONS: Use the KERNEL_LIBSVM function to evaluate each of the
% kernels you mentioned. Then run the line below to save the results to a
% .mat file.
k = @(x,x2) kernel_poly(x, x2, 1);
results.linear = kernel_libsvm(X, Y, Xtest, Ytest, k);% ERROR RATE OF LINEAR KERNEL GOES HERE
k = @(x,x2) kernel_poly(x, x2, 2);
results.quadratic =kernel_libsvm(X, Y, Xtest, Ytest, k); % ERROR RATE OF QUADRATIC KERNEL GOES HERE
k = @(x,x2) kernel_poly(x, x2, 3);
results.cubic =kernel_libsvm(X, Y, Xtest, Ytest, k); % ERROR RATE OF CUBIC KERNEL GOES HERE
k = @(x,x2) kernel_gaussian(x, x2, 20);
results.gaussian =kernel_libsvm(X, Y, Xtest, Ytest, k); % ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
k = @(x,x2) kernel_intersection(x, x2);
results.intersect =kernel_libsvm(X, Y, Xtest, Ytest, k); % ERROR RATE OF INTERSECTION KERNEL GOES HERE

% Makes a bar chart showing the errors of the different algorithms.
algs = fieldnames(results);
for i = 1:numel(algs)
    y(i) = results.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Kernel');
ylabel('Test Error');
title('Kernel Comparisons');

print -djpeg -r72 plot_1.jpg;

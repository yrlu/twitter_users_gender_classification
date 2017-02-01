
%% Plots/submission for Perceptron portion, Question 2.

%% Put your written answers here.
clear all
close all
answers{1} = 'a. Passive aggressive converges more quickly; b.constant update has the lower training error; c.passive aggressive has the lower testing error; d. it is worse than SVM linear d.';

save('problem_2_answers.mat', 'answers');

%% Load and process data
load ../data/breast-cancer-data.mat;

%Include bias column of all 1's
Xtest = [ones(size(Xtest, 1),1) Xtest];
Xtrain = [ones(size(Xtrain, 1),1) Xtrain];


% INSTRUCTIONS: Use the averaged_perceptron_train function to train model
% using learning rate of 1.0
numPasses = 8; %Do not change

update_fnc = @(x,y,w) update_constant(x,y,w,1.0);
[w_avg err] = averaged_perceptron_train(Xtrain, Ytrain, update_fnc, numPasses);
results.w_const  = w_avg;%Averaged W for constant learning rate 1.0
results.train_err_const = err;%Error vector for constant learning rate 1.0
results.test_err_const = perceptron_error(Xtest,Ytest, w_avg);
disp(['final averaged error:  ', num2str(results.test_err_const)]);


[w_avg err] = averaged_perceptron_train(Xtrain, Ytrain, @update_passive_aggressive, numPasses);
results.w_pa = w_avg;%Averaged W for passive aggressive update
results.train_err_pa = err;%Error vector for passive aggressive update
results.test_err_pa = perceptron_error(Xtest,Ytest, w_avg);
disp(['final passive aggressive error:  ', num2str(results.test_err_pa)])


% Makes a plot comparing learning curves of different updating methods
algs = {'const', 'pa'};
legend_names = {'Constant', 'Passive aggressive'};
figure
for i = 1:numel(algs)
    plot(results.(['train_err_' algs{i}]));
    hold all
end
hold off
xlabel('Iteration');
ylabel('Training Error');
title('Learning Curve');
legend(legend_names)

print -djpeg -r72 plot_2.1.jpg;


%Comparison with SVM
[results.test_err_svm info] = kernel_libsvm(Xtrain, Ytrain, Xtest, Ytest, @(x1,x2)kernel_poly(x1,x2,1));

% Makes a bar chart showing the errors of the different algorithms.
figure
algs = [algs 'svm'];
legend_names = [legend_names 'Linear SVM'];
for i = 1:numel(algs)
    y(i) = results.(['test_err_' algs{i}]);
end
bar(y);
set(gca,'XTickLabel', legend_names);
ylabel('Test Error');
title('Algorithm Comparisons');

print -djpeg -r72 plot_2.2.jpg;


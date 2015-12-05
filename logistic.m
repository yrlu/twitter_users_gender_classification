function [  predicted_label] = logistic( train_x, train_y, test_x, test_y )
    model = train(train_y, sparse(train_x), ['-s 0 -q', 'col']);
    [predicted_label, accuracy, prob_estimates] = predict(test_y, sparse(test_x), model, ['-q', 'col']);
end


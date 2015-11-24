% Author: Max Lu
% Date: Nov 23

% Input: 
%   ypred = [log_out nn_1 nn_2 rf_1 rf_2]
%   thres = [thres_log, thres_nn, thres_rf]

% Description:
%   In this function we do cascading prediction by the RAW data
%   produced by the 3 classifiers. The thresholds are applied to each of
%   the absolute value of the outputs. 
%   If the first classifier satisfys the desirable confidence, then just
%   output the label. 
%   else turn to the next.
%   If all the three thresholds are not passed, then leave a -1 instead.

function [Yhat] = acc_cascading(ypred, thres)
    Yhat = [];
    for i = 1:size(ypred,1)
%        if abs(ypred(i,1)) > thres(1)
%            Yhat = [Yhat;ypred(i,1)<0];
       if abs(ypred(i,2)-ypred(i,3)) > thres(2)
           Yhat = [Yhat;ypred(i,2)>ypred(i,3)];
       elseif abs(ypred(i,4)) > thres(2)
           Yhat = [Yhat;ypred(i,4)<0];
       else 
           Yhat = [Yhat;-1];
       end
    end
end
% Author: Max Lu
% Date: Dec 5

function [data1, data2] = gen_data_separate(data, proportion)
    data1 = data(1:end*proportion, :);
    data2 = data(end*proportion+1:end,:);
end
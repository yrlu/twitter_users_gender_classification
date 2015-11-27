% Author: Max Lu
% Date: Nov 27



%% 


% Read data:

genders_train=dlmread('train/genders_train.txt');
images_train=dlmread('train/images_train.txt');
image_features_train=dlmread('train/image_features_train.txt');
words_train=dlmread('train/words_train.txt');


image_features_test=dlmread('test/image_features_test.txt');
images_test=dlmread('test/images_test.txt');
words_test=dlmread('test/words_test.txt');

%%
% Read stop words list:
stopwords=importdata('texts/stopwords.txt');

% Read 5000 words list 
words = readtable('texts/voc-top-5000.txt','Delimiter',' ');
words_index = table2array(words(:,1));
words_index = [0;words_index];
words_str = table2array(words(:,2));
words_str = ['0.0', words_str];

% wordsleft = words;

remove = ones(size(words_str,1),1);

for i = 1:size(stopwords,1)
    stopword = stopwords(i);
    i
    for j = 1:size(words_str,1)
%         words_str(j)
        if strcmp(words_str(j), stopword)
            remove(j) = 0;
        end
    end
end

dlmwrite('texts/stopwords_index.txt', remove, '\n');


%% words stemming:


stem_map = containers.Map;
mapping_idx = [1];
count = 1;

for i = 1:size(words_str,1)
    stem = porter_stemmer(words_str(i));
    stem_map(stem) = count;
    mapping_idx = [count];
end






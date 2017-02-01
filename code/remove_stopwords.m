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
% words_index = [0;words_index];
words_str = table2array(words(:,2));
% words_str = ['0.0', words_str];
size(words_str)
% wordsleft = words;


%% words stemming & stopwords:
% % Read stop words list:
% stopwords=importdata('texts/stopwords.txt');
% 
% % Read 5000 words list 
% words = readtable('texts/voc-top-5000.txt','Delimiter',' ');
% words_index = table2array(words(:,1));
% words_index = [0;words_index];
% words_str = table2array(words(:,2));
% words_str = ['0.0', words_str];

% wordsleft = words;

words_train=dlmread('train/words_train.txt');
words_test=dlmread('test/words_test.txt');


words = readtable('texts/voc-top-5000.txt','Delimiter',' ');

words.Properties.VariableNames = {'id', 'word'};
cellPatients = {0, '0.0'};
% T2 = cell2table(cellPatients(1:end,:));
% cellPatients = {'LastName','Gender','Age','Height','Weight',...
%     'Smoker','Systolic','Diastolic';
%     'Edwards','Male',42,70,158,0,116,83;
%     'Falk','Female',28,62,125,1,120,71};
T2 = cell2table(cellPatients);
T2.Properties.VariableNames = {'id', 'word'};

words = [T2;words];


words_index = table2array(words(:,1));
words_str = table2array(words(:,2));





remove = ones(size(words_str,1),1);

for i = 1:size(stopwords,1)
    stopword = stopwords(i);
%     i
    for j = 1:size(words_str,1)
%         words_str(j)
        if strcmp(words_str(j), stopword)
            remove(j) = 0;
        end
    end
end

dlmwrite('texts/stopwords_index.txt', remove, '\n');

%%



stem_map = containers.Map('KeyType','char', 'ValueType','any')
mapping_idx = [];
mapping_words = {};
count = 1;

word_stem = cell(3996,1);
for i = 1:size(words_str,1)
    words_str(i)
    stem = char(porter_stemmer(char(words_str(i))));
    stem
    
    if ~stem_map.isKey(stem)
        stem_map(stem) = count;
        word_stem{count} = stem;
        count = count + 1;
    else
        stem_map(stem)
    end
    mapping_idx = [mapping_idx;stem_map(stem)];
    words_str(i) = {stem};
end


% word_stem

words_stem_train = zeros(size(words_train,1), mapping_idx(end));

for i = 1:size(words_train,1)
%     i
    for j = 1:size(words_train,2);
%         j
        if remove(j) 
            words_stem_train(i, mapping_idx(j)) =  words_stem_train(i, mapping_idx(j)) + words_train(i,j);
        end
    end
end


save('train/words_stem_train.mat', 'words_stem_train');

words_stem_test = zeros(size(words_test,1), mapping_idx(end));

for i = 1:size(words_test,1)
%     i
    for j = 1:size(words_test,2);
%         j
        if remove(j)
            words_stem_test(i, mapping_idx(j)) =  words_stem_test(i, mapping_idx(j)) + words_test(i,j);
        end
    end
end

save('test/words_stem_test.mat', 'words_stem_test');

save('texts/word_stem.mat', 'word_stem');




% mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims create_ngram_tree.cpp
% mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims get_tree_featurespace.cpp
%mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims get_tree_vocab.cpp
%mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims tree_cull_vocab.cpp
%mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims tree_cull_vocab_v.cpp
%mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims fastkern.cpp 
mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims train_fastnb.cpp
mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims predict_fastnb.cpp
% mex CFLAGS="\$CFLAGS -Wall -Werror -O3 -std=c99" CPPFLAGS="\$CPPFLAGS -std=c++98" -largeArrayDims fast_featureselect.cpp

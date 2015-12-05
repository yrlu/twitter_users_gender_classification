/*
 * create_ngram_tree.cpp
 *
 *  Copyright 2013 Gareth Cross
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *	Created on: 2013-11-26
 *		Author: Gareth
 */

#include "TreeNode.hpp"

void build_ngrams(std::vector<std::string>& ngrams, const std::vector<std::string>& unigrams, int N);
bool clean_predicate(char c);
std::string trim_whitespace(const std::string& str, const std::string& whitespace = " \t\n\r");

/**
 *  @brief [tree,size] = create_ngrame_tree(texts, N) creates ngram tree from review texts
 *  
 *  @param texts Cell array of cell arrays, each filled with unigrams
 *  @param N Type of grams to generate (2, 3, etc...)
 *  @return tree Tree structure generated
 *  @param size Size of the vocabulary
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 1 && nrhs != 2)
    {
        mexErrMsgIdAndTxt(  "create_ngram_tree:invalidNumInputs",
                            "One or two inputs required.");
    }
    else if (nlhs != 1 && nlhs != 2)
    {
        mexErrMsgIdAndTxt(  "create_ngram_tree:invalidNumOutputs",
                            "One or two outputs required.");
    }
    else if(!mxIsCell(prhs[0]))
    {
        mexErrMsgIdAndTxt(  "create_ngram_tree:inputNotStruct",
                            "Input must be a cell array.");
    }
    else if ((nrhs==2) && !IS_SCALAR(prhs[1]))
    {
        mexErrMsgIdAndTxt(  "create_ngram_tree:inputNotScalar",
                            "Second input must be a scalar");
    }
    
    TreeNode * tree = 0;
    
    size_t num_observations = mxGetNumberOfElements(prhs[0]);
    if (num_observations == 0) {
        mexErrMsgIdAndTxt("create_ngram_tree:invalidInput",
                          "There must be more than 0 observations!");
    }
    
    double integral = 2.0;  //  default is 2 (bigrams)
    if (nrhs == 2) {
        const double gram_count_input = mxGetScalar(prhs[1]);
        if (std::modf(gram_count_input, &integral) != 0.0) {
            mexErrMsgIdAndTxt("create_ngram_tree:invalidInput",
                              "Bigram count must be an integer!");
        }
    }
    
    const int gram_count = static_cast<int>(integral);
    
    //mexPrintf("Running ngram tree on %i observations\n", num_observations);
    //mexPrintf("Building tree...\n");
    //mexEvalString("drawnow;");  //  force flush of IO
    
    //  iterate over observations
    for (size_t i=0; i < num_observations; i++)
    {
        mxArray * cell = mxGetCell(prhs[0], i);
        
        if (!mxIsCell(cell)) {
            if (tree) {
                delete tree;
            }
            mexErrMsgIdAndTxt("create_ngram_tree:invalidInput",
                              "Input must be a cell array.");
        }
        
        size_t num_unigrams = mxGetNumberOfElements(cell);
        
        //  iterate over unigrams for this observation
        std::vector<std::string> unigrams_cleaned;
        unigrams_cleaned.reserve(num_unigrams);
        
        for (size_t n=0; n < num_unigrams; n++)
        {
            mxArray * gram_cell = mxGetCell(cell, n);
            
            if (!mxIsChar(gram_cell)) {
                if (tree) {
                    delete tree;
                }
                mexErrMsgIdAndTxt(  "create_ngram_tree:invalidInput",
                                    "Input cell arrays must contain strings.");
            }
            
            char * cstr = mxArrayToString(gram_cell);

            //  convert ngram to cpp string
            std::string unigram_string = std::string(cstr);
            mxFree(cstr);
            
            //  make lowercase
            std::transform(unigram_string.begin(), unigram_string.end(), unigram_string.begin(), ::tolower);
            
            //  remove everything except alphanumerics + spaces and tabs
            unigram_string.erase(std::remove_if(unigram_string.begin(), unigram_string.end(), clean_predicate), unigram_string.end());
            
            //  trim starting and ending whitespace
            unigram_string = trim_whitespace(unigram_string);
            
            if (unigram_string.empty()) {
                continue;
            }
            
            //mexPrintf("extracted unigram: %s\n", unigram_string.c_str());
            unigrams_cleaned.push_back(unigram_string);
        }
        
        //  build bigrams
        std::vector <std::string> bigrams;
        build_ngrams(bigrams, unigrams_cleaned, gram_count);
        
        //  append to tree
        for (std::vector <std::string> :: iterator it = bigrams.begin(); it != bigrams.end(); it++) {
            if (!tree) {
                tree = new TreeNode(*it);
            }
            tree->append_increment(*it, static_cast<int>(i));
        }
    }
    
    int col=0;
    if (tree) {
        tree->assign_columns(col);      //  lazy - traverse tree to assign column values
    }
    
    //mexPrintf("Done building tree, %lu instances.\n", tree->count_observations());
    
    //  debug
    //std::string left = tree->leftmost_token();
    //std::string right = tree->rightmost_token();
    //mexPrintf("Leftmost term: %s, rightmost term: %s\n", left.c_str(), right.c_str());
    
    //  pass back tree
    if (tree)
    {
        plhs[0] = tree->create_mex_struct();
        
        if (nlhs == 2) {
            plhs[1] = mxCreateDoubleScalar((double)tree->count_nodes());
        }
        
        //  cleanup
        delete tree;
    }
    else
    {
        //  no tree was created, not enough unigrams - pass back logical false
        plhs[0] = mxCreateLogicalScalar(false);
        if (nlhs == 2) {
            plhs[1] = mxCreateDoubleScalar(0);
        }
    }
}

void build_ngrams(std::vector<std::string>& ngrams, const std::vector<std::string>& unigrams, int N)
{
    std::string str;
    for (size_t i=0; i < unigrams.size(); i++)
    {
        str="";
        if (i+N-1 < unigrams.size())
        {
            for (size_t j=i; j < i+N; j++)
            {
                str.append(unigrams[j]);
                if (j != i+N-1) {
                    str.append(" ");
                }
            }
            ngrams.push_back(str);
        }
    }
}

bool clean_predicate(char c)
{
    if (isalnum(c) || c==' ' || c=='\t')
    {
        return false;
    }
    return true;
}

std::string trim_whitespace(const std::string& str, const std::string& whitespace)
{
    const size_t strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content
    
    const size_t strEnd = str.find_last_not_of(whitespace);
    const size_t strRange = strEnd - strBegin + 1;
    
    return str.substr(strBegin, strRange);
}

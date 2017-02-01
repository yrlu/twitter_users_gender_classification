/*
 * tree_cull_vocab_v.c
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
 *	Created on: 12/2/2013
 *		Author: Gareth Cross
 */

#include "TreeNode.hpp"

/**
 *  @brief [idx_1, idx_2] = tree_cull_vocab(tree_1, vocab) determine the intersection of
 *  a tree and a cell array of vocab.
 *  
 *  @param tree_1 First tree
 *  @param vocab Cell array of strings
 *
 *  @return idx_1 Indices into the feature-space of tree 1
 *  @return idx_2 Indices into the feature-space of the vocab array
 */
void mexFunction( int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 2)
    {
        mexErrMsgIdAndTxt("tree_cull_vocab_v:invalidNumInputs",
                          "Two inputs required.");
    }
    else if (nlhs != 2)
    {
        mexErrMsgIdAndTxt("tree_cull_vocab_v:invalidNumOutputs",
                          "Two outputs required.");
    }
    else if(!mxIsStruct(prhs[0]))
    {
        mexErrMsgIdAndTxt("tree_cull_vocab_v:inputNotStruct",
                          "First inputs must be struct.");
    }
    else if (!mxIsCell(prhs[1]))
    {
        mexErrMsgIdAndTxt("tree_cull_vocab_v:inputNotStruct",
                          "Second input must be cell array of strings.");
    }
    
    size_t num_vocab = mxGetNumberOfElements(prhs[1]);
    if (num_vocab == 0) {
        mexErrMsgIdAndTxt("create_ngram_tree_v:invalidInput",
                          "There must be more than 0 vocabulary terms.");
    }
    
    //  load left trees
    TreeNode * lTree = new TreeNode();
    try {
        lTree->load_mex_struct(prhs[0]);
    }
    catch (std::exception& e) {
        delete lTree;
        mexErrMsgIdAndTxt("tree_cull_vocab_v:invalidInput", "Failed to parse tree: %s", e.what());
    }
    
    //mexPrintf("Loading %lu terms from right tree. Culling...\n", num_vocab);
    //mexEvalString("drawnow;");  //  force flush of IO
    
    //  stick results in logical array
    mwSize dim = num_vocab;
    mxArray * log_right = mxCreateLogicalArray(1, &dim);    //  dimension of num_vocab
    
    dim = lTree->count_nodes();
    mxArray * log_left = mxCreateLogicalArray(1, &dim); //  dimension of left tree vocab
    
    //mexPrintf("%lu terms in left tree\n", dim);
    
    mxLogical * l_data = static_cast<mxLogical*>(mxGetData(log_left));  //  initialized to false
    mxLogical * r_data = static_cast<mxLogical*>(mxGetData(log_right));
    
    size_t kept=0;
    for (mwIndex i = 0; i < num_vocab; i++)
    {
        //  pull string
        mxArray * cell = mxGetCell(prhs[1], i);
        
        if (!mxIsChar(cell)) {
            delete lTree;   //  make sure tree is destroyed
            mexErrMsgIdAndTxt("create_ngram_tree_v:invalidInput",
                              "Second input must be a cell array of strings.");
        }
        
        char * cstr = mxArrayToString(cell);
        std::string term = std::string(cstr);
        mxFree(cstr);
        
        bool found;
        int order = 0;
        TreeNode * n = lTree->find(term, found, order);
        if (found) {
            l_data[n->column] = 1;
            r_data[i] = 1;
            kept++;
        }
    }
    //mexPrintf("%lu terms in tree\n", c1);
    
    //mexPrintf("%lu terms remain.\n", kept);
    
    //  return logical array
    plhs[0] = log_left;
    plhs[1] = log_right;
    
    //  cleanup
    delete lTree;
}

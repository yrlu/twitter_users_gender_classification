/*
 * tree_cull_vocab.cpp
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
 *	Created on: 11/27/2013
 *		Author: Gareth Cross
 */
 
#include "TreeNode.hpp"

/**
 *  @brief [idx_1, idx_2] = tree_cull_vocab(tree_1, tree_2) determine the intersection of trees
 *  
 *  @param tree_1 First tree
 *  @param tree_2 Second tree
 *
 *  @return idx_1 Indices into the feature-space of tree 1
 *  @return idx_2 Indices into the feature-space of tree 2
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 2)
    {
        mexErrMsgIdAndTxt("tree_cull_vocab:invalidNumInputs",
                          "Two inputs required.");
    }
    else if (nlhs != 2)
    {
        mexErrMsgIdAndTxt("tree_cull_vocab:invalidNumOutputs",
                          "Two outputs required.");
    }
    else if(!mxIsStruct(prhs[0]) || !mxIsStruct(prhs[1]))
    {
        mexErrMsgIdAndTxt("tree_cull_vocab:inputNotStruct",
                          "Both inputs must be structs.");
    }
    
    //  load both trees
    TreeNode * lTree = new TreeNode();
    TreeNode * rTree = new TreeNode();
    try {
        lTree->load_mex_struct(prhs[0]);
        rTree->load_mex_struct(prhs[1]);
    }
    catch (std::exception& e) {
        delete lTree;
        delete rTree;
        mexErrMsgIdAndTxt("tree_cull_vocab:invalidInput", "Failed to parse tree: %s", e.what());
    }
    
    //  get vocab of trees
    //  Note this is inefficient since all we need is the node count, not the actual voca
    std::vector<std::string> rVocab;
    rTree->extract_tokens(rVocab);
    
    //mexPrintf("Loading %lu terms from right tree. Culling...\n", rVocab.size());
    //mexEvalString("drawnow;");  //  force flush of IO
    
    //  stick results in logical array
    mwSize dim = rVocab.size();
    mxArray * log_right = mxCreateLogicalArray(1, &dim);
    
    dim = lTree->count_nodes();
    mxArray * log_left = mxCreateLogicalArray(1, &dim); //  dimension of left tree vocab

    mxLogical * l_data = static_cast<mxLogical*>(mxGetData(log_left));
    mxLogical * r_data = static_cast<mxLogical*>(mxGetData(log_right));
    
    size_t count=0,kept=0;
    for (std::vector<std::string> :: iterator i = rVocab.begin(); i != rVocab.end(); i++)
    {
        bool found;
        int order =0;
        TreeNode * n = lTree->find(*i, found, order);
        
        if (found) {
            l_data[n->column] = 1;
            r_data[count] = 1;
            kept++;
        }
        
        count++;
    }
    
    //mexPrintf("%lu terms remain.\n", kept);
    
    //  return logical array
    plhs[0] = log_left;
    plhs[1] = log_right;
    
    //  cleanup
    delete lTree;
    delete rTree;
}

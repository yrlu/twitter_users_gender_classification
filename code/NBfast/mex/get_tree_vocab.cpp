/*
 * get_tree_vocab.cpp
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
 *  @brief [vocab] = get_tree_vocab(tree) extract ngrams from tree
 *  
 *  @param tree Tree generated with create_ngram_tree
 *  
 *  @return vocab Cell array of strings in alphabetical order
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 1)
    {
        mexErrMsgIdAndTxt(  "get_tree_vocab:invalidNumInputs",
                            "One input required.");
    }
    else if (nlhs != 1)
    {
        mexErrMsgIdAndTxt(  "get_tree_vocab:invalidNumOutputs",
                            "One output required.");
    }
    else if(!mxIsStruct(prhs[0]))
    {
        mexErrMsgIdAndTxt(  "get_tree_vocab:inputNotStruct",
                            "Input must be a struct.");
    }
    
    TreeNode * tree = new TreeNode();
    try {
        tree->load_mex_struct(prhs[0]);
    }
    catch (std::exception& e) {
        delete tree;
        mexErrMsgIdAndTxt("get_tree_vocab:invalidInput", "Failed to parse tree: %s", e.what());
    }
    
    //  pull all the vocab out of the tree
    std::vector<std::string> vocab;
    tree->extract_tokens(vocab);
    //mexPrintf("%lu vocabulary terms loaded\n", vocab.size());
    
    //  put the vocab in a cell array
    mwSize vocab_size = vocab.size();
    plhs[0] = mxCreateCellArray(1, &vocab_size);
    
    mwIndex idx=0;
    for (std::vector<std::string> :: iterator i = vocab.begin(); i != vocab.end(); i++)
    {
        mxArray * vstring = mxCreateString(i->c_str());
        mxSetCell(plhs[0], idx++, vstring);
    }
    
    //  cleanup
    delete tree;
}

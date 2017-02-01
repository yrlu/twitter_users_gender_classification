/*
 * get_tree_featurespace.cpp
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
 *  @brief X = get_tree_featurespace(tree, N, size) extract feature-space matrix from tree
 *  
 *  @param tree Tree generated with create_ngram_tree
 *  @param N Number of observations used to generate tree
 *  @param size Size of the vocabulary stored in the tree
 *  
 *  @return X Matrix of observations, where the columns are features
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 3)
    {
        mexErrMsgIdAndTxt(  "get_tree_featurespace:invalidNumInputs",
                            "Three inputs required.");
    }
    else if (nlhs != 1)
    {
        mexErrMsgIdAndTxt(  "get_tree_featurespace:invalidNumOutputs",
                            "One output required.");
    }
    else if(!mxIsStruct(prhs[0]))
    {
        mexErrMsgIdAndTxt(  "get_tree_featurespace:inputNotStruct",
                            "First tnput must be a struct.");
    }
    else if (!IS_SCALAR(prhs[1]) || !IS_SCALAR(prhs[2]))
    {
        mexErrMsgIdAndTxt("get_tree_featurespace:inputNotNumeric",
                          "Second/third inputs must be scalar dimensions.");
    }
    
    mwSize m = static_cast<mwSize>(mxGetScalar(prhs[1]));   //  number of rows in X (# of observations!)
    mwSize n = static_cast<mwSize>(mxGetScalar(prhs[2]));   //  number of columns in X (size of vocab!)
    
    if (!m || !n) {
        mexErrMsgIdAndTxt("get_tree_featurespace:inputNotFinite",
                          "Second/third inputs must be non zero!");
    }
    
    TreeNode * tree = new TreeNode();
    try {
        tree->load_mex_struct(prhs[0]);
    }
    catch (std::exception& e) {
        delete tree;
        mexErrMsgIdAndTxt("get_tree_featurespace:invalidInput", "Failed to parse tree: %s", e.what());
    }

    //  get # of observations, this is the number of filled spaces in the sparse matrix (avoid mxRealloc later)
    size_t num_observations = tree->count_observations();
    
    //mexPrintf("Generating %lu x %lu sparse matrix, with %lu instances (%.5f %%)\n", m, n, num_observations, num_observations*100.0f / ((float)m*n));
    
    //  fill the sparse array
    mxArray * sparse = mxCreateSparse(m,n,num_observations,mxREAL);
    
    double * d = mxGetPr(sparse);
    mwIndex * ir = mxGetIr(sparse);
    mwIndex * jc = mxGetJc(sparse);
        
    mwIndex idx=0, start_col=0;
    tree->extract_features(d, ir, jc, idx, start_col);
    jc[start_col] = idx;  //  append last index
    
    //mexPrintf("Traversed tree, loading %lu instances into %lu columns.\n", idx, start_col);
    
    //  return sparse array
    plhs[0] = sparse;
    
    //  cleanup
    delete tree;
}


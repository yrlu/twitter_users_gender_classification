/*
 * train_fastnb.cpp
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
 *	Created on: 12/4/2013
 *		Author: Gareth Cross
 */

extern "C" {
    #include "mex.h"
    #include "stdint.h"
    #include "string.h"
}

#include <vector>
#include <cmath>

/**
 *  @brief model = train_fastnb(X,Y,labels) trains a multinomial Naive Bayes classifier
 *  @param X MxN Matrix of observations, where columns are features
 *  @param Y Mx1 Matrix of labels
 *  @param labels 1xK Matrix of labels, where K is the number of unique labels in Y
 *  @note Uses laplace smoothing to handle missing features.
 *  @see predict_fastnb
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 3)
    {
        mexErrMsgIdAndTxt("train_fastnb:invalidNumInputs",
                          "Three inputs required.");
    }
    else if (nlhs != 1)
    {
        mexErrMsgIdAndTxt("train_fastnb:invalidNumOutputs",
                          "One output required.");
    }
    else if(!mxIsSparse(prhs[0]))
    {
        mexErrMsgIdAndTxt("train_fastnb:inputNotSparse",
                          "First input must be sparse.");
    }
    else if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]))
    {
        mexErrMsgIdAndTxt("train_fastnb:inputNotNumeric",
                          "All inputs must be double");
    }
    
    const mxArray * X = prhs[0];
    const mxArray * Y = prhs[1];
    const mxArray * labels = prhs[2];
    
    const mwSize M = mxGetM(X);
    const mwSize N = mxGetN(X);
    const mwSize K = mxGetN(labels);
    
    //  check dimensions
    if (mxGetM(Y) != M) {
        mexErrMsgIdAndTxt("train_fastnb:invalidInputDimensions",
                          "X and Y must have same number of rows.");
    }
    if (mxGetN(Y) != 1 || !M) {
        mexErrMsgIdAndTxt("train_fastnb:invalidInputDimensions",
                          "Y should be a M x 1 vector (M > 0)");
    }
    if (mxGetM(labels) != 1 || !K) {
        mexErrMsgIdAndTxt("train_fastnb:invalidInputDimensions",
                          "Labels should be 1 x K vector (K > 0)");
    }
    
    //  feature space data
    const mwIndex * xIR = mxGetIr(X);
    const mwIndex * xJC = mxGetJc(X);
    const double * xVals = mxGetPr(X);
    
    //  label data
    const double * yData = static_cast<double *>( mxGetData(Y) );
    const double * labelData = static_cast<double *>( mxGetData(labels) );
    
    //  storage for class probabilities
    mxArray * classProbs = mxCreateNumericMatrix(0,0,mxSINGLE_CLASS, mxREAL);
    mxSetM(classProbs, 1);
    mxSetN(classProbs, K);
    float * cp = static_cast<float *>( mxMalloc(K * sizeof(float)) );
    mxSetData(classProbs, cp);
    
    //  storage for word probabilities
    mxArray * wordCounts = mxCreateNumericMatrix(0,0,mxSINGLE_CLASS, mxREAL);
    mxSetM(wordCounts, K);
    mxSetN(wordCounts, N);
    float * wp = static_cast<float * >( mxMalloc(K * N * sizeof(float)) );
    memset(wp, 0, K * N * sizeof(float));
    mxSetData(wordCounts, wp);
    
    //  storage for labels (output)
    mxArray * labelsOut = mxDuplicateArray(prhs[2]);
    
    //  create matrix of occurrences of labels
    std::vector <bool> Z(M*K, false);
    std::vector <int> Z_sum(K, 0);
    
#define Z_ACC(i,k)  Z[(i)*K + (k)]
    
    for (mwIndex i=0; i < M; i++)
    {
        for (mwIndex k=0; k < K; k++)
        {
            if (yData[i] == labelData[k])
            {
                Z_ACC(i, k) = true;
                Z_sum[k]++;
            }
        }
    }
    
    //  calculate class log probabilities
    for (mwIndex k=0; k < K; k++)
    {
        float p = logf(Z_sum[k]) - logf(M * 1.0f);
        cp[k] = p;
    }

    //  number of words in each class
    std::vector<float> words_in_class(K, 0.0f);
    
    //  step 1: count words
    
    //  iterate over features/words
    for (mwIndex n=0; n < N; n++)
    {
        mwIndex rowStart = xJC[n];
        mwIndex rowEnd = xJC[n+1];
        
        if (rowStart == rowEnd) {
            //  empty column, this feature never occurs
            continue;
        }
        rowEnd--;
        
        //  iterate over observations
        for (mwIndex i=rowStart; i <= rowEnd; i++)
        {
            const mwIndex m = xIR[i];                                         //  row index
            const float word_count = static_cast<float>( xVals[i] );          //  count of feature n in observation m
            
            //  iterate over classes
            for (mwIndex k=0; k < K; k++)
            {
                if (Z_ACC(m, k) == true)    //  observation belongs to class k
                {
                    wp[n*K + k] += word_count;                    //  increment # obvs word n in class k
                    words_in_class[k] += word_count;              //  increment total # of words in class k
                    break;
                }
            }
        }
    }
    
    //  step 2: convert into log probabilities
    
    //  iterate over words
    for (mwIndex n=0; n < N; n++)
    {
        //  iterate over classes
        for (mwIndex k=0; k < K; k++)
        {
            //  use laplace smoothing
            wp[n*K + k] = logf(wp[n*K + k] + 1.0f) - logf(N + words_in_class[k]);
        }
    }
    
    //  generate output structure...
    const char * fieldNames[] = {"classes", "class_prob", "feature_prob"};
    plhs[0] = mxCreateStructMatrix(1, 1, 3, fieldNames);
    
    mxSetField(plhs[0], 0, fieldNames[0], labelsOut);
    mxSetField(plhs[0], 0, fieldNames[1], classProbs);
    mxSetField(plhs[0], 0, fieldNames[2], wordCounts);
}


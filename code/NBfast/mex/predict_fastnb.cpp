/*
 * predict_fastnb.cpp
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
 *  @brief Y = predict_fastnb(model, X) performs predictions using a multinomial Naive Bayes classifier
 *  @param model A model generated using train_fastnb
 *  @param X MxN Matrix of observations, where columns are observations (ie. X is transposed!)
 *  @return Y NxK Matrix of log probabilities, where K is the number of labels in the model
 *  @note This method is about 100x faster than Matlab's NaiveBayes.predict
 *  @see train_fastnb
 */
void mexFunction( int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 2)
    {
        mexErrMsgIdAndTxt("predict_fastnb:invalidNumInputs",
                          "Two inputs required.");
    }
    else if (nlhs != 1)
    {
        mexErrMsgIdAndTxt("predict_fastnb:invalidNumOutputs",
                          "One output required.");
    }
    else if(!mxIsStruct(prhs[0]))
    {
        mexErrMsgIdAndTxt("predict_fastnb:inputNotStruct",
                          "First input must be struct.");
    }
    else if (!mxIsDouble(prhs[1]) || !mxIsSparse(prhs[1]))
    {
        mexErrMsgIdAndTxt("predict_fastnb:inputNotSparse",
                          "Second input must be sparse double!");
    }
    
    const mxArray * model = prhs[0];
    const mxArray * X = prhs[1];    //  this input should be transposed!
    
    const mxArray * featureProbs = mxGetField(model, 0, "feature_prob");
    const mxArray * classProbs = mxGetField(model, 0, "class_prob");
    const mxArray * classes = mxGetField(model, 0, "classes");
    if (!featureProbs || !classProbs || !classes) {
        mexErrMsgIdAndTxt("predict_fast_nb:missingField", "One or more fields is missing from the model!");
    }
    
    //  TODO: Check data types and sizes here!
    float * wp = static_cast<float*>( mxGetData(featureProbs) );
    float * cp = static_cast<float*>( mxGetData(classProbs) );
    
    const mwSize M = mxGetM(X);
    const mwSize N = mxGetN(X);
    
    //  check dimensions
    if (M == 0 || N == 0) {
        mexErrMsgIdAndTxt("predict_fastnb:invalidInputDimensions", "X must have non-zero dimensions");
    }
    
    if (mxGetN(featureProbs) != M) {
        mexErrMsgIdAndTxt("predict_fastnb:invalidInputDimensions",
                          "X does not have the same number of features as the model. Did you remember to transpose X?");
    }
    
    //  feature space data
    const mwIndex * xIR = mxGetIr(X);
    const mwIndex * xJC = mxGetJc(X);
    const double * xVals = mxGetPr(X);
    
    //  label data
    const mwSize K = mxGetN(classes);
    
    //  output is N x K, with probabilities of each class
    mxArray * yProbs = mxCreateNumericMatrix(N,K, mxDOUBLE_CLASS, mxREAL);
    mxSetM(yProbs, N);
    mxSetN(yProbs, K);
    double * yp = static_cast<double *>( mxMalloc(N * K * sizeof(double)) );
    mxSetData(yProbs, yp);
    
    //  iterate over observations, which are in columns now
    for (mwIndex n=0; n < N; n++)
    {
        mwIndex rowStart = xJC[n];
        mwIndex rowEnd = xJC[n+1];
        
        if (rowStart == rowEnd) {
            //  observation with no features - predict class probabilies
            for (mwIndex k=0; k < K; k++)
            {
                yp[k*N + n] = static_cast<double>(cp[k]);
            }
            continue;
        }
        rowEnd--;
        
        std::vector<double> log_prob_n_given_k(K, 0.0);     //  log probability of seeing observation n, given class k
        std::vector<double> log_prob_n_and_k(K, 0.0);
        
        //  iterate over features, which are in rows now
        for (mwIndex i=rowStart; i <= rowEnd; i++)
        {
            const mwIndex m = xIR[i];               //   m is index of feature
            const double word_count = xVals[i];     //   number of times word appears
            
            //  get probability of word for each class
            for (mwIndex k=0; k < K; k++)
            {
                double log_p_word = static_cast<double>(wp[m*K + k]);       //  P(word)
                log_p_word *= word_count;                                   //  P(word)^count
                log_prob_n_given_k[k] += log_p_word;                        //  P(word1)^count1 * P(word2)^count2 * ...
            }
        }
        
        //  calculate probability of observation itself using handy formula
        for (mwIndex k=0; k < K; k++) {
            log_prob_n_and_k[k] = cp[k] + log_prob_n_given_k[k];    //  log(P(c) * P(n|c))
        }
        
        double sum=0.0f;
        for (mwIndex k=1; k < K; k++) {
            sum += exp(log_prob_n_and_k[k] - log_prob_n_and_k[0]);
        }
        sum = log_prob_n_and_k[0] + log(1.0f + sum);
        
        //  calculate class probabilities
        for (mwIndex k=0; k < K; k++) {
            yp[k*N + n] = exp( static_cast<double>( cp[k] ) + log_prob_n_given_k[k] - sum );
        }
    }
    
    //  done
    plhs[0] = yProbs;
}

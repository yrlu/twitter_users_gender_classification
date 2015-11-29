/*
 * fast_featureselect.cpp
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

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 2)
    {
        mexErrMsgIdAndTxt("fast_featureselect:invalidNumInputs",
                          "Two inputs required.");
    }
    else if (nlhs != 1)
    {
        mexErrMsgIdAndTxt("fast_featureselect:invalidNumOutputs",
                          "One output required.");
    }
    else if(!mxIsClass(prhs[0], "FeatureSelector"))
    {
        mexErrMsgIdAndTxt("fast_featureselect:inputNotClass",
                          "First input must of type FeatureSelector.");
    }
    else if (!mxIsDouble(prhs[1]) || !mxIsSparse(prhs[1]))
    {
        mexErrMsgIdAndTxt("fast_featureselect:inputNotSparse",
                          "Second input must be sparse double!");
    }
    
    const mxArray * featureSelector = prhs[0];
    const mxArray * X = prhs[1];
    
    const mwIndex * xJC = mxGetJc(X);
    const double * xVals = mxGetPr(X);
    
    const mwSize M = mxGetM(X);
    const mwSize N = mxGetN(X);
    
    if (M != 1) {
        mexErrMsgIdAndTxt("fast_featureselect:invalidDimensions",
                          "X must be a row vector!");
    }
    
    const mxArray * bns_scores = mxGetProperty(featureSelector, 0, "bns_scores");
    const mxArray * feat_ind = mxGetProperty(featureSelector, 0, "feat_ind");
    const mxArray * bns_max = mxGetProperty(featureSelector, 0, "bns_max");
    const mxArray * binary = mxGetProperty(featureSelector, 0, "binary");
    const mxArray * scale = mxGetProperty(featureSelector, 0, "scale");
    
    if ((mxGetClassID(feat_ind) != mxUINT32_CLASS) || (mxGetClassID(bns_scores) != mxUINT16_CLASS)) {
        mexErrMsgIdAndTxt("fast_featureselect:invalidFormat",
                          "Feature indices must be 32bit, scores must be 16bit!");
    }
    
    const char * scale_str = mxArrayToString(scale);
    if (!scale_str || !strlen(scale_str)) {
        mexErrMsgIdAndTxt("fast_featureselect:invalidOption", "Unspecified scale string!");
    }
    const bool scale_by_bns = (strcmp(scale_str, "bns") == 0);
    
    const bool binary_mode = mxIsLogicalScalarTrue(binary);
    const double bns_max_val = mxGetScalar(bns_max);
    
    const mwSize numidx = mxGetN(feat_ind);
    const uint32_t * indices = static_cast<uint32_t*>( mxGetData(feat_ind) );
    
    const mwSize numScores = mxGetN(bns_scores);
    const uint16_t * scores = static_cast<uint16_t*>( mxGetData(bns_scores) );
    
    if (numidx != numScores) {
        mexErrMsgIdAndTxt("fast_featureselect:invalidDimensions", "bns_scores and feat_ind must have identical dimensions!");
    }
    
    //mexPrintf("%lu value\n", numidx);
    
    mxArray * Xout = mxCreateSparse(1, numidx, numidx, mxREAL);     //  not really sparse but it will do
    
    double * xreal = mxGetPr(Xout);    //  outputs
    mwIndex * outIR = mxGetIr(Xout);
    mwIndex * outJC = mxGetJc(Xout);
    
    mwIndex pos=0, i;
    for (i=0; i < numidx; i++)
    {
        const mwIndex n = static_cast<mwIndex> ( indices[i] ) - 1;      //  convert to C index
        if (n > N) {
            mxDestroyArray(Xout);
            mexErrMsgIdAndTxt("fast_featureselect:invalidIndex", "Invalid index in feat_ind");
        }
        
        const mwIndex rowStart = xJC[n];
        const mwIndex rowEnd = xJC[n+1];
        
        outJC[i] = pos;
        
        if (rowStart != rowEnd)
        {   //  there is a value here
            
            double val = xVals[rowStart];
            
            if (binary_mode) {
                //  use binary threshold
                val = 1.0;
            }
        
            if (scale_by_bns) {
                //  convert from 16 bit compressed state
                val *= (static_cast<double>(scores[i]) * bns_max_val * 0.0000152590219);
            }
            
            //  save value
            if (val != 0.0)
            {
                xreal[pos] = val;
                outIR[pos] = 0;
                outJC[i] = pos;
                
                pos++;
            }
        }
    }
    outJC[i] = pos;
    
    plhs[0] = Xout;
}

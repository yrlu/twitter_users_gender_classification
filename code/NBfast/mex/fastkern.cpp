/*
 * fastkern.cpp
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

extern "C" {
    #include "mex.h"
    #include "stdint.h"
    #include "string.h"
}

#include <cmath>
#include <algorithm>

#define MIN(a,b)    (((a) <= (b)) ? (a) : (b))
#define MAX(a,b)    (((a) >= (b)) ? (a) : (b))

/**
 *  @brief K = fastkern(X1,X2) computes a fast intersection kernel of two matrices
 *  
 *  @param X1 DxM sparse matrix (Observations in columns)
 *  @param X2 DxN sparse matrix (Observations in columns)
 *  @param K MxN full matrix, where K(i,j) is the intersection of column i of X2, and column j of X1
 *  
 *  @note fastkern() stores the result as 16bit unsigned integers, demanding that the inputs
 *  be positive, and that the sum of any intersection not exceed 65535.
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 2)
    {
        mexErrMsgIdAndTxt("fastkern:invalidNumInputs",
                          "Two inputs required.");
    }
    else if (nlhs != 1)
    {
        mexErrMsgIdAndTxt("fastkern:invalidNumOutputs",
                          "One output required.");
    }
    else if(!mxIsSparse(prhs[0]) || !mxIsSparse(prhs[1]) || !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
    {
        mexErrMsgIdAndTxt("fastkern:inputNotSparse",
                          "Both inputs must be sparse doubles.");
    }
    
    const mxArray * X1 = prhs[0];
    const mxArray * X2 = prhs[1];
    
    //  get dimensions
    
    const mwSize N = mxGetN(X1);  //  N of result
    const mwSize M = mxGetN(X2);  //  M of result
    const mwSize D = mxGetM(X1);
    
    if (D != mxGetM(X2)) {
        mexErrMsgIdAndTxt("fastkern:invalidInputDimensions",
                          "Inputs must have the same number of rows.");
    }
    
    //  create a full matrix in this implementation
    //  use 16 bit to reduce storage requirements
    mxArray * K = mxCreateNumericMatrix(0, 0, mxUINT16_CLASS, mxREAL);
    uint16_t * kdata = (uint16_t *)mxMalloc(M * N * sizeof(uint16_t));
    memset(kdata, 0, M * N * sizeof(uint16_t));
    
    const double * x1_real = mxGetPr(X1); //  input 1
    const mwIndex * x1_ir = mxGetIr(X1);
    const mwIndex * x1_jc = mxGetJc(X1);
    
    const double * x2_real = mxGetPr(X2); //  input 2
    const mwIndex * x2_ir = mxGetIr(X2);
    const mwIndex * x2_jc = mxGetJc(X2);
    
    const size_t buf_size = D * sizeof(double);
    double * buf = (double *)mxMalloc(buf_size);
    
    //  iterate over columns in result
    for (mwIndex j=0; j < N; j++)
    {
        //  this column of X1 is empty
        if (x1_jc[j] == x1_jc[j+1])
            continue;
        
        //  unpack column j of X1
        memset(buf, 0, buf_size);
        for (mwIndex d = x1_jc[j]; d <= x1_jc[j+1]-1; d++) {
            buf[ x1_ir[d] ] = x1_real[d];
        }
        
        //  rows in result
        for (mwIndex i=0; i < M; i++)
        {
            //  space (i,j) in K is the j'th column of X2, i'th column of X1
            
            mwIndex sr2 = x2_jc[i];         //  start indices for columns i,i+1 of X2
            mwIndex er2 = x2_jc[i+1];
            
            if (sr2 == er2) {
                continue;                   //  this column of X2 is empty
            }
            er2--;                          //  calc end index
            
            double sum = 0;
            for (; sr2 <= er2; sr2++)
            {
                const mwIndex row2 = x2_ir[sr2];    //  row index of X2
                
                sum += MIN(buf[row2], x2_real[sr2]);
            }
            long s = (long)sum;
            kdata[j*M + i] = uint16_t(s & 0xFFFF);
        }
    }
    
    mxFree(buf);

    //  pass memory to matlab
    mxSetData(K, kdata);
    mxSetM(K, M);
    mxSetN(K, N);
    
    //  done
    plhs[0] = K;
}




/*
 * TreeNode.hpp
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
 *     Purpose: Created for CIS520 final project/competition
 *		Author: Gareth Cross
 */

#ifndef TreeNode_hpp
#define TreeNode_hpp

extern "C" {
    #include "mex.h"
    #include "string.h"
    #include "stdint.h"
    #include "ctype.h"
}

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <locale>
#include <cmath>

#define IS_SCALAR(x) (mxIsDouble(x) && !mxIsComplex(x) && (mxGetM(x)==1) && (mxGetN(x)==1))

const char * mexFieldNames[] = {"token", "left", "right", "observations", "counts", "column"};

struct TreeNode
{
    std::string token;
    
    TreeNode * left;
    TreeNode * right;
    
    TreeNode * parent;
    
    std::map <int,int> observations;
    
    int column;
    
    TreeNode(const std::string& T = "") : token(T), left(0), right(0), column(0) {
        
    }
    
    TreeNode(const TreeNode& rhs) {
        token = rhs.token;
        left = rhs.left;    //  careful with copying pointers here...
        right = rhs.right;
        parent = rhs.parent;
        observations = rhs.observations;
        column = rhs.column;
    }
    
    ~TreeNode() {
        if (left) {
            delete left; left=0;
        }
        if (right) {
            delete right; right=0;
        }
    }
    
    mxArray * create_mex_struct()
    {
        mxArray * out = mxCreateStructMatrix(1, 1, 6, mexFieldNames);
        
        //  token
        mxArray * token_str = mxCreateString(token.c_str());
        mxSetField(out, 0, mexFieldNames[0], token_str);
        
        //  build observations and count arrays
        mxArray * obvs = mxCreateNumericMatrix(0, 0, mxINT32_CLASS, mxREAL);
        mxArray * counts = mxCreateNumericMatrix(0, 0, mxINT32_CLASS, mxREAL);
        
        int32_t * obvs_data = (int32_t *)mxMalloc(observations.size() * sizeof(int32_t));
        int32_t * count_data = (int32_t *)mxMalloc(observations.size() * sizeof(int32_t));
        
        mwIndex idx=0;
        for (std::map<int, int> ::iterator i = observations.begin(); i != observations.end(); i++) {
            
            obvs_data[idx] = static_cast<int32_t>(i->first);
            count_data[idx] = static_cast<int32_t>(i->second);
            
            idx++;
        }
        
        //  store in numeric matrices (matlab manages this memory now, not us)
        mxSetM(obvs, 1);
        mxSetN(obvs, idx);
        mxSetData(obvs, obvs_data);
        
        mxSetM(counts, 1);
        mxSetN(counts, idx);
        mxSetData(counts, count_data);
        
        //  set children
        if (left) {
            mxSetField(out, 0, mexFieldNames[1], left->create_mex_struct());
        }
        if (right) {
            mxSetField(out, 0, mexFieldNames[2], right->create_mex_struct());
        }
        
        //  indices
        mxSetField(out, 0, mexFieldNames[3], obvs);
        mxSetField(out, 0, mexFieldNames[4], counts);
        
        //  column
        mxArray * scalar = mxCreateNumericMatrix(0,0, mxINT32_CLASS, mxREAL);
        int32_t *col = (int32_t *)mxMalloc(sizeof(int32_t));
        *col = static_cast<int32_t>(column + 1);    //  add one going to matlab space
        
        mxSetM(scalar, 1);
        mxSetN(scalar, 1);
        mxSetData(scalar, col);
        
        mxSetField(out, 0, mexFieldNames[5], scalar);
        
        return out;
    }
    
    void load_mex_struct(const mxArray * in) throw(std::runtime_error)
    {
        if (!mxIsStruct(in)) {
            throw std::runtime_error("Not a struct");
        }
        
        if (mxGetNumberOfFields(in) != 6) {
            throw std::runtime_error("Wrong number fields on struct");
        }
        
        //  token
        mxArray * t = mxGetField(in, 0, mexFieldNames[0]);
        char * cstr = mxArrayToString(t);
        token = std::string(cstr);
        mxFree(cstr);
        
        //  left and right
        mxArray * l = mxGetField(in, 0, mexFieldNames[1]);
        if (l) {
            this->left = new TreeNode();
            this->left->parent = this;
            this->left->load_mex_struct(l);
        }
        
        mxArray * r = mxGetField(in, 0, mexFieldNames[2]);
        if (r) {
            this->right = new TreeNode();
            this->right->parent = this;
            this->right->load_mex_struct(r);
        }
        
        //  observations
        mxArray * obvs = mxGetField(in, 0, mexFieldNames[3]);
        mxArray * counts = mxGetField(in, 0, mexFieldNames[4]);
        
        //  don't support anything but int32 here...
        if (mxGetClassID(obvs) != mxINT32_CLASS || mxGetClassID(counts) != mxINT32_CLASS) {
            throw std::runtime_error("Wrong data type (not INT32) for observations");
        }
        
        int32_t * obvs_data = static_cast<int32_t*>(mxGetData(obvs));
        int32_t * counts_data = static_cast<int32_t*>(mxGetData(counts));
        
        if (!obvs_data || !counts_data) {
            throw std::runtime_error("Observations are nil!");
        }
        
        //  stick in map (assume row vectors here...)
        for (mwIndex j=0; j < mxGetN(obvs); j++) {
            this->observations[(int)obvs_data[j]] = (int)counts_data[j];
        }
        
        //  column
        mxArray *col = mxGetField(in, 0, mexFieldNames[5]);
        if (mxGetClassID(col) != mxINT32_CLASS) {
            throw std::runtime_error("Wrong data type (not INT32) for column");
        }
        
        //  get value
        int32_t * col_data = static_cast<int32_t*>(mxGetData(col));
        this->column = (int)(*col_data) - 1;    //  subtract one to go back from matlab space
    }
    
    TreeNode * find(const std::string& t, bool& found, int& order) const
    {
        order = strcmp(token.c_str(), t.c_str());
        if (order == 0)
        {
            found = true;
        }
        else if (order > 0)
        {
            if (!left) {
                found = false;
            } else {
                return left->find(t, found, order);
            }
        }
        else
        {
            if (!right) {
                found = false;
            } else {
                return right->find(t, found, order);
            }
        }
        
        return (TreeNode *)this;
    }
     
    void assign_columns(int& col)
    {
        if (left) {
            left->assign_columns(col);
        }
        
        column = col;
        col = col + 1;
        
        if (right) {
            right->assign_columns(col);
        }
    }
    
    void append_increment(const std::string& t, int observation)
    {
        bool found = false;
        int order = 0;
        
        TreeNode * node = find(t, found, order);
        if (found)
        {
            node->observations[observation] += 1;
        }
        else
        {
            TreeNode * child = new TreeNode(t);
            child->parent = this;
            child->observations[observation] = 1;
            
            if (order > 0) {
                node->left = child;
            } else {
                node->right = child;
            }
        }
    }
    
    void extract_tokens(std::vector<std::string>& tokens)
    {
        if (left) {
            left->extract_tokens(tokens);
        }
        
        tokens.push_back(token);
        
        if (right) {
            right->extract_tokens(tokens);
        }
    }
    
    TreeNode * leftmost_node()
    {
        TreeNode * node = this;
        while (node->left) {
            node = node->left;
        }
        return node;
    }
    
    size_t count_observations()
    {
        size_t count = observations.size();
        
        if (left) {
            count += left->count_observations();
        }
        
        if (right) {
            count += right->count_observations();
        }
        
        return count;
    }
    
    size_t count_nodes()
    {
        size_t count = 1;
        
        if (left) {
            count += left->count_nodes();
        }
        
        if (right) {
            count += right->count_nodes();
        }
        
        return count;
    }
    
    void extract_features( double * real, mwIndex * rows, mwIndex * cols, mwIndex& idx, mwIndex& start_col)
    {
        //  begin with left, so that start_col is appropriately incremented
        //  bottom-left node is column 0, and so on
        
        if (left) {
            left->extract_features(real, rows, cols, idx, start_col);
        }
        
        //  mark the start of this column
        cols[start_col] = idx;
        
        //  now do this node
        for (std::map<int,int>::iterator i = observations.begin(); i != observations.end(); i++)
        {
            rows[idx] = static_cast<mwIndex>(i->first);
            real[idx] = static_cast<double>(i->second);
            
            idx++;
        }
        
        //  translate right
        start_col = start_col + 1;
        
        //  then do right
        if (right) {
            right->extract_features(real, rows, cols, idx, start_col);
        }
    }
};

#endif

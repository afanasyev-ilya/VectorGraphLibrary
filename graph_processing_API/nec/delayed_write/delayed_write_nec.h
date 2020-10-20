#pragma once

#include "architectures.h"
#include <cfloat>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DelayedWriteNEC
{
public:
    int int_vec_reg[VECTOR_LENGTH];
    float flt_vec_reg[VECTOR_LENGTH];
    double dbl_vec_reg[VECTOR_LENGTH];

    inline void init();

    inline void init(int *_array, int _val);
    inline void init(float *_array, float _val);
    inline void init(double *_array, double _val);

    inline void start_write(int *_data, int _val, int _vector_index);
    inline void start_write(float *_data, float _val, int _vector_index);
    inline void start_write(double *_data, double _val, int _vector_index);

    inline void finish_write_max(int *_data, int _idx);
    inline void finish_write_max(float *_data, int _idx);
    inline void finish_write_max(double *_data, int _idx);

    inline void finish_write_min(int *_data, int _idx);
    inline void finish_write_min(float *_data, int _idx);
    inline void finish_write_min(double *_data, int _idx);

    inline void finish_write_sum(int *_data, int _idx);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "delayed_write_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

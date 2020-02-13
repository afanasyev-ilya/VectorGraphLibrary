#pragma once

#include "../../../architectures.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DelayedWriteNEC
{
private:
    int int_vec_reg[VECTOR_LENGTH];
    float flt_vec_reg[VECTOR_LENGTH];
    double dbl_vec_reg[VECTOR_LENGTH];

public:
    inline void init();

    inline void start_write(int *_data, int _val, int _vector_index);
    inline void start_write(float *_data, float _val, int _vector_index);
    inline void start_write(double *_data, double _val, int _vector_index);

    inline void finish_write_max(int *_data, int _idx);
    inline void finish_write_max(float *_data, int _idx);
    inline void finish_write_max(double *_data, int _idx);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "delayed_write_nec.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

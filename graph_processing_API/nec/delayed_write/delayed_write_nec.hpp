/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::init()
{
    /*#pragma _NEC vreg(int_vec_reg)
    #pragma _NEC vreg(flt_vec_reg)
    #pragma _NEC vreg(dbl_vec_reg)*/

    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        int_vec_reg[i] = 0;
        flt_vec_reg[i] = 0;
        dbl_vec_reg[i] = 0;
    }

    pack_int_1 = new int[256];
    pack_int_2 = new int[256];

    pack_int_1_to_flt = (float*)pack_int_1;
    pack_int_2_to_flt = (float*)pack_int_2;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::init(int *_array, int _val)
{
    for(int i = 0; i < VECTOR_LENGTH; i++)
        int_vec_reg[i] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::init(float *_array, float _val)
{
    for(int i = 0; i < VECTOR_LENGTH; i++)
        flt_vec_reg[i] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::init(double *_array, double _val)
{
    for(int i = 0; i < VECTOR_LENGTH; i++)
        dbl_vec_reg[i] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::start_write(int *_data, int _val, int _vector_index)
{
    int_vec_reg[_vector_index] = _val;
}

void DelayedWriteNEC::start_write(float *_data, float _val, int _vector_index)
{
    flt_vec_reg[_vector_index] = _val;
}

void DelayedWriteNEC::start_write(double *_data, double _val, int _vector_index)
{
    dbl_vec_reg[_vector_index] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::finish_write_max(int *_data, int _idx)
{
    int old_val = _data[_idx];

    #pragma _NEC unroll(VECTOR_LENGTH)
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(int_vec_reg[i] > old_val)
            old_val = int_vec_reg[i];
    _data[_idx] = old_val;
}

void DelayedWriteNEC::finish_write_max(float *_data, int _idx)
{
    float old_val = _data[_idx];

    #pragma _NEC unroll(VECTOR_LENGTH)
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(flt_vec_reg[i] > old_val)
            old_val = flt_vec_reg[i];
    _data[_idx] = old_val;
}

void DelayedWriteNEC::finish_write_max(double *_data, int _idx)
{
    double old_val = _data[_idx];

    #pragma _NEC unroll(VECTOR_LENGTH)
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(dbl_vec_reg[i] > old_val)
            old_val = dbl_vec_reg[i];
    _data[_idx] = old_val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::finish_write_min(int *_data, int _idx)
{
    int old_val = _data[_idx];

    #pragma _NEC unroll(VECTOR_LENGTH)
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(int_vec_reg[i] < old_val)
            old_val = int_vec_reg[i];

    if(old_val < _data[_idx])
        _data[_idx] = old_val;
}

void DelayedWriteNEC::finish_write_min(float *_data, int _idx)
{
    float old_val = FLT_MAX;

    #pragma _NEC unroll(VECTOR_LENGTH)
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(flt_vec_reg[i] < old_val)
            old_val = flt_vec_reg[i];

    if(old_val < _data[_idx])
        _data[_idx] = old_val;
}

void DelayedWriteNEC::finish_write_min(double *_data, int _idx)
{
    double old_val = _data[_idx];

    #pragma _NEC unroll(VECTOR_LENGTH)
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(dbl_vec_reg[i] < old_val)
            old_val = dbl_vec_reg[i];

    if(old_val < _data[_idx])
        _data[_idx] = old_val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::finish_write_sum(int *_data, int _idx)
{
    int sum = 0;

    #pragma _NEC unroll(VECTOR_LENGTH)
    for(int i = 0; i < VECTOR_LENGTH; i++)
        sum += int_vec_reg[i];

    _data[_idx] = sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


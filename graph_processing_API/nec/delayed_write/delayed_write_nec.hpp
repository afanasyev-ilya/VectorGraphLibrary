/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::init()
{
    #pragma _NEC vreg(int_vec_reg)
    #pragma _NEC vreg(flt_vec_reg)
    #pragma _NEC vreg(dbl_vec_reg)

    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        int_vec_reg[i] = 0;
        flt_vec_reg[i] = 0;
        dbl_vec_reg[i] = 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::start_write(int *_data, int _val, int _vector_index)
{
    int_vec_reg[_vector_index] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::start_write(float *_data, float _val, int _vector_index)
{
    flt_vec_reg[_vector_index] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::start_write(double *_data, double _val, int _vector_index)
{
    dbl_vec_reg[_vector_index] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::finish_write_max(int *_data, int _idx)
{
    int old_val = _data[_idx];
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(int_vec_reg[i] > old_val)
            old_val = int_vec_reg[i];
    _data[_idx] = old_val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::finish_write_max(float *_data, int _idx)
{
    float old_val = _data[_idx];
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(flt_vec_reg[i] > old_val)
            old_val = flt_vec_reg[i];
    _data[_idx] = old_val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DelayedWriteNEC::finish_write_max(double *_data, int _idx)
{
    double old_val = _data[_idx];
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(dbl_vec_reg[i] > old_val)
            old_val = dbl_vec_reg[i];
    _data[_idx] = old_val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

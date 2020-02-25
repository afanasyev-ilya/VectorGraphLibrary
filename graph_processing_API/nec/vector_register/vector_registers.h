/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define NEC_REGISTER_INT(name, value)\
int reg_##name[VECTOR_LENGTH];\
for(int i = 0; i < VECTOR_LENGTH; i++)\
{\
    reg_##name[i] = 0;\
}

#define NEC_REGISTER_FLT(name, value)\
float reg_##name[VECTOR_LENGTH];\
for(int i = 0; i < VECTOR_LENGTH; i++)\
{\
    reg_##name[i] = 0;\
}

#define NEC_REGISTER_DBL(name, value)\
double reg_##name[VECTOR_LENGTH];\
for(int i = 0; i < VECTOR_LENGTH; i++)\
{\
    reg_##name[i] = 0;\
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T register_sum_reduce(_T reg_name[VECTOR_LENGTH])
{
    _T sum = 0;
    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
        sum += reg_name[i];
    return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T register_max_reduce(_T reg_name[VECTOR_LENGTH])
{
    _T max = 0;
    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(max < reg_name[i])
            reg_name[i] = max;
    return max;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

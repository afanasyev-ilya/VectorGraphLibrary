/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define NEC_REGISTER_INT(name, value)\
int reg_##name[VECTOR_LENGTH];\
for(int i = 0; i < VECTOR_LENGTH; i++)\
{\
    reg_##name[i] = value;\
}

#define NEC_REGISTER_FLT(name, value)\
float reg_##name[VECTOR_LENGTH];\
for(int i = 0; i < VECTOR_LENGTH; i++)\
{\
    reg_##name[i] = value;\
}

#define NEC_REGISTER_DBL(name, value)\
double reg_##name[VECTOR_LENGTH];\
for(int i = 0; i < VECTOR_LENGTH; i++)\
{\
    reg_##name[i] = value;\
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
    _T max = std::numeric_limits<_T>::min();
    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(reg_name[i] > max)
            max = reg_name[i];
    return max;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T register_min_reduce(_T reg_name[VECTOR_LENGTH])
{
    _T min = std::numeric_limits<_T>::max();
    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
        if(reg_name[i] < min)
            min = reg_name[i];
    return min;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

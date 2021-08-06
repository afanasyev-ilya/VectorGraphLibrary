#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ASL_CALL( CallInstruction ) { \
    int res = CallInstruction; \
    if(res != ASL_ERROR_OK) { \
        string error_str = "";\
        if(res == ASL_ERROR_LIBRARY_UNINITIALIZED)\
            error_str = "ASL_ERROR_LIBRARY_UNINITIALIZED";\
        else if(res == ASL_ERROR_RANDOM_INVALID)\
            error_str = "ASL_ERROR_RANDOM_INVALID";\
        else if(res == ASL_ERROR_RANDOM_INCOMPATIBLE_CALL)\
            error_str = "ASL_ERROR_RANDOM_INCOMPATIBLE_CALL";\
        else if(res == ASL_ERROR_ARGUMENT)\
            error_str = "ASL_ERROR_ARGUMENT";\
        else if(res == ASL_ERROR_MEMORY)\
            error_str = "ASL_ERROR_MEMORY";\
        else if(res == ASL_ERROR_SORT_INVALID)\
            error_str = "ASL_ERROR_SORT_INVALID";\
        else if(res == ASL_ERROR_SORT_INCOMPATIBLE_CALL)\
        	error_str = "ASL_ERROR_SORT_INCOMPATIBLE_CALL";\
        else \
            error_str = "unknown";\
        printf("ASL error: %s at call \"" #CallInstruction "\"\n", error_str.c_str()); \
    } \
}\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ASLRandomGenerator::ASLRandomGenerator()
{
    ASL_CALL(asl_library_initialize());
    ASL_CALL(asl_random_create(&hnd, ASL_RANDOMMETHOD_AUTO));
    //asl_uint32_t seed = 100;
    //ASL_CALL(asl_random_initialize(hnd, 100000, &seed));
    ASL_CALL(asl_random_distribute_uniform(hnd));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void ASLRandomGenerator::generate_array_of_random_uniform_values(_T *_array, asl_int_t _size, _T _max_val)
{
    /* generic implementation  */
    throw "ERROR: unsupported type";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ASLRandomGenerator::~ASLRandomGenerator()
{
    ASL_CALL(asl_random_destroy(hnd));
    ASL_CALL(asl_library_finalize());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void ASLRandomGenerator::generate_array_of_random_uniform_values<float>(float *_array, asl_int_t _size, float _max_val)
{
    ASL_CALL(asl_random_generate_s(hnd, _size, _array));
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        _array[i] = abs(_array[i]) * _max_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void ASLRandomGenerator::generate_array_of_random_uniform_values<double>(double *_array, asl_int_t _size, double _max_val)
{
    ASL_CALL(asl_random_generate_d(hnd, _size, _array));
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        _array[i] = abs(_array[i]) * _max_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void ASLRandomGenerator::generate_array_of_random_uniform_values<int>(int *_array, asl_int_t _size, int _max_val)
{
    ASL_CALL(asl_random_generate_uniform_bits(hnd, _size, (asl_uint32_t*)_array));

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        _array[i] = abs(_array[i]) % _max_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

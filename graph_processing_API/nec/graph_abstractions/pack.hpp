#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T1, typename _T2>
inline void vgl_unpack(VGL_PACK_TYPE _val, _T1 &_a, _T1 &_b)
{
    int a = (int)((_val & 0xFFFFFFFF00000000LL) >> 32);
    int b = (int)(_val & 0xFFFFFFFFLL);

    _a = *(_T1*)&b;
    _b = *(_T2*)&a;
}

inline void vgl_unpack(VGL_PACK_TYPE _val, int &_a, int &_b)
{
    _a = (int)((_val & 0xFFFFFFFF00000000LL) >> 32);
    _b = (int)(_val & 0xFFFFFFFFLL);
}

inline void vgl_unpack(VGL_PACK_TYPE _val, float &_a, int &_b)
{
    int a = (int)((_val & 0xFFFFFFFF00000000LL) >> 32);
    int b = (int)(_val & 0xFFFFFFFFLL);

    _a = *(float*)&a;
    _b = *(int*)&b;
}

inline void vgl_unpack(VGL_PACK_TYPE _val, float &_a, float &_b)
{
    int a = (int)((_val & 0xFFFFFFFF00000000LL) >> 32);
    int b = (int)(_val & 0xFFFFFFFFLL);

    _a = *(float*)&a;
    _b = *(float*)&b;
}

inline void vgl_unpack(VGL_PACK_TYPE _val, int &_a, float &_b)
{
    int a = (int)((_val & 0xFFFFFFFF00000000LL) >> 32);
    int b = (int)(_val & 0xFFFFFFFFLL);

    _a = *(int*)&a;
    _b = *(float*)&b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T1, typename _T2>
inline void vgl_pack(VGL_PACK_TYPE &_val, _T1 _a, _T1 _b)
{
    int a = *(int*)&_b;
    int b = *(int*)&_a;
    _val = ((VGL_PACK_TYPE)a) << 32 | b;
}

inline void vgl_pack(VGL_PACK_TYPE &_val, int _a, float _b)
{
    int a = *(int*)&_a;
    int b = *(int*)&_b;
    _val = ((VGL_PACK_TYPE)a) << 32 | b;
}

inline void vgl_pack(VGL_PACK_TYPE &_val, float _a, int _b)
{
    int a = *(int*)&_a;
    int b = *(int*)&_b;
    _val = ((VGL_PACK_TYPE)a) << 32 | b;
}

inline void vgl_pack(VGL_PACK_TYPE &_val, float _a, float _b)
{
    int a = *(int*)&_a;
    int b = *(int*)&_b;
    _val = ((VGL_PACK_TYPE)a) << 32 | b;
}

inline void vgl_pack(VGL_PACK_TYPE &_val, int _a, int _b)
{
    int a = _a;
    int b = _b;
    _val = ((VGL_PACK_TYPE)a) << 32 | b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T1, typename _T2>
void GraphAbstractionsNEC::pack_vertices_arrays(VerticesArray<VGL_PACK_TYPE> &_packed_data,
                                                VerticesArray<_T1> &_first,
                                                VerticesArray<_T2> &_second)
{
    VGL_PACK_TYPE *packed_ptr = _packed_data.get_ptr();
    _T1 *first_ptr = _first.get_ptr();
    _T2 *second_ptr = _second.get_ptr();

    int size = _packed_data.size();
    if(size != _first.size())
        throw "Error in GraphAbstractionsNEC::pack_vertices_arrays : non-equal sizes of input arrays";
    if(size != _second.size())
        throw "Error in GraphAbstractionsNEC::pack_vertices_arrays : non-equal sizes of input arrays";

    Timer tm;
    tm.start();
    #pragma _NEC cncall
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for schedule(static)
    for(int src_id = 0; src_id < size; src_id++)
    {
        vgl_pack(packed_ptr[src_id], first_ptr[src_id], second_ptr[src_id]);
    }
    tm.end();

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Pack vertices arrays", size, sizeof(VGL_PACK_TYPE) + 2*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T1, typename _T2>
void GraphAbstractionsNEC::unpack_vertices_arrays(VerticesArray<VGL_PACK_TYPE> &_packed_data,
                                                  VerticesArray<_T1> &_first,
                                                  VerticesArray<_T2> &_second)
{
    VGL_PACK_TYPE *packed_ptr = _packed_data.get_ptr();
    _T1 *first_ptr = _first.get_ptr();
    _T2 *second_ptr = _second.get_ptr();

    cout << typeid(_first[0]).name() << endl;
    cout << typeid(_second[0]).name() << endl;

    int size = _packed_data.size();
    if(size != _first.size())
        throw "Error in GraphAbstractionsNEC::unpack_vertices_arrays : non-equal sizes of input arrays";
    if(size != _second.size())
        throw "Error in GraphAbstractionsNEC::unpack_vertices_arrays : non-equal sizes of input arrays";

    Timer tm;
    tm.start();
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for schedule(static)
    for(int src_id = 0; src_id < size; src_id++)
    {
        vgl_unpack<_T1, _T2>(packed_ptr[src_id], first_ptr[src_id], second_ptr[src_id]);
    }
    tm.end();

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Unpack vertices arrays", size, sizeof(VGL_PACK_TYPE) + 2*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

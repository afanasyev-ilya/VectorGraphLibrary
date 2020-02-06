/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::estimate_nec_thresholds()
{
    nec_all_cores_threshold_vertex = 0;
    nec_single_core_threshold_vertex = 0;

    for(int i = 0; i < this->vertices_count - 1; i++)
    {
        if(i < 10)
            cout << outgoing_ptrs[i+1] - outgoing_ptrs[i] << endl;

        int current_size = outgoing_ptrs[i+1] - outgoing_ptrs[i];
        int next_size = outgoing_ptrs[i+2] - outgoing_ptrs[i+1];
        if((current_size > NEC_ALL_CORES_THRESHOLD_VALUE) && (next_size <= NEC_ALL_CORES_THRESHOLD_VALUE))
        {
            nec_all_cores_threshold_vertex = i + 1;
        }
        if((current_size > NEC_SINGLE_CORE_THRESHOLD_VALUE) && (next_size <= NEC_SINGLE_CORE_THRESHOLD_VALUE))
        {
            nec_single_core_threshold_vertex = i + 1;
        }
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

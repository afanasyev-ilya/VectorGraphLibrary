#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../lp_constants.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_lp_wrapper(VectorCSRGraph &_graph, int *_labels, int &_iterations_count,
                    GpuActiveConditionType _gpu_active_condition_type, int _max_iterations);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

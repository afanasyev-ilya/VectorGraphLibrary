#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void select_device(int _num)
{
    SAFE_CALL(cudaSetDevice(_num));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


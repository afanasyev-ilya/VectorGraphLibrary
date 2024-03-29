#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SnapShotStruct
{
    int u;
    int v;
    int stage;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SCC::seq_tarjan_kernel(VGL_Graph &_graph,
                            int _root,
                            VerticesArray<int> &_disc,
                            VerticesArray<int> &_low,
                            stack<int> &_st,
                            VerticesArray<bool> &_stack_member,
                            VerticesArray<int> &_components)
{
    static int time = 0;
    static int comp = 0;

    stack<SnapShotStruct> snapshotStack;
    SnapShotStruct currentSnapshot;
    currentSnapshot.u = _root;
    currentSnapshot.v = -1;
    currentSnapshot.stage = 0;

    snapshotStack.push(currentSnapshot);

    while (!snapshotStack.empty()) // do DFS without recursion
    {
        currentSnapshot = snapshotStack.top();
        snapshotStack.pop();

        switch (currentSnapshot.stage)
        {
            case 0:
            {
                int u = currentSnapshot.u;
                _disc[u] = _low[u] = ++time;
                _st.push(u);
                _stack_member[u] = true;

                SnapShotStruct retSnapshot;
                retSnapshot.u = u;
                retSnapshot.v = 0;
                retSnapshot.stage = 2;
                snapshotStack.push(retSnapshot);

                const int connections_count = _graph.get_outgoing_connections_count(u);
                for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
                {
                    int v = _graph.get_outgoing_edge_dst(u, edge_pos);

                    if (_disc[v] == -1)
                    {
                        currentSnapshot.u = u;
                        currentSnapshot.v = v;
                        currentSnapshot.stage = 1;
                        snapshotStack.push(currentSnapshot);

                        SnapShotStruct newSnapshot;
                        newSnapshot.u = v;
                        newSnapshot.v = 0;
                        newSnapshot.stage = 0;
                        snapshotStack.push(newSnapshot);
                    }
                    else if (_stack_member[v] == true)
                    {
                        _low[u] = min(_low[u], _disc[v]);
                    }
                }
            }
                break;
            case 1:
            {
                int u = currentSnapshot.u;
                int v = currentSnapshot.v;
                _low[u] = min(_low[u], _low[v]);
            }
                break;
            case 2:
            {
                int u = currentSnapshot.u;
                int w = 0;  // To store stack extracted vertices
                if (_low[u] == _disc[u])
                {
                    while (_st.top() != u)
                    {
                        w = _st.top();
                        _stack_member[w] = false;
                        _st.pop();
                        _components[w] = comp;
                    }
                    w = _st.top();
                    _components[w] = comp;
                    _stack_member[w] = false;
                    _st.pop();

                    comp++;
                }
            }
                break;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double SCC::seq_tarjan(VGL_Graph &_graph, VerticesArray<int> &_components)
{
    // allocate memory for Tarjan's algorithm computations
    Timer tm;
    tm.start();
    VerticesArray<int>disc(_graph);
    VerticesArray<int>low(_graph);
    VerticesArray<bool>stack_member(_graph);
    stack<int> st;

    // Initialize disc and low, and stackMember arrays
    for (int i = 0; i < _graph.get_vertices_count(); i++)
    {
        disc[i] = -1;
        low[i] = -1;
        stack_member[i] = false;
    }

    // run algorithm
    for (int root = 0; root < _graph.get_vertices_count(); root++)
    {
        if (disc[root] == -1)
        {
            seq_tarjan_kernel(_graph, root, disc, low, st, stack_member, _components);
        }
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    performance_stats.print_algorithm_performance_stats("SCC (Sequential Tarjan)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

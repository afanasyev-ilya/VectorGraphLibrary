#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SnapShotStruct
{
    int u;
    int v;
    int stage;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SCC::seq_tarjan_kernel(VectCSRGraph &_graph,
                            int _root,
                            VerticesArrayNec<int> &_disc,
                            VerticesArrayNec<int> &_low,
                            stack<int> &_st,
                            VerticesArrayNec<bool> &_stack_member,
                            VerticesArrayNec<int> &_components)
{
    UndirectedCSRGraph *outgoing_graph_ptr = _graph.get_outgoing_graph_ptr();
    LOAD_UNDIRECTED_CSR_GRAPH_DATA((*outgoing_graph_ptr));

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

                const long long edge_start = vertex_pointers[u];
                const int connections_count = vertex_pointers[u + 1] - vertex_pointers[u];
                for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
                {
                    long long int global_edge_pos = edge_start + edge_pos;
                    int v = adjacent_ids[global_edge_pos];

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

void SCC::seq_tarjan(VectCSRGraph &_graph, VerticesArrayNec<int> &_components)
{
    cout << "tarjan seq started" << endl;

    // allocate memory for Tarjan's algorithm computations

    VerticesArrayNec<int>disc(_graph, SCATTER);
    VerticesArrayNec<int>low(_graph, SCATTER);
    VerticesArrayNec<bool>stack_member(_graph, SCATTER);
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

    _components.print();
    _graph.save_to_graphviz_file<int>("scc.gv", _components);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

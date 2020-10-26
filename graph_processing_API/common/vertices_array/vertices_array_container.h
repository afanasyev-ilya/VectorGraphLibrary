#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VerticesArrayContainer
{
private:
    TraversalDirection direction;
    char *pointer;
    int element_size;
public:
    VerticesArrayContainer(char *_pointer, int _element_size, TraversalDirection _direction)
    {
        pointer = _pointer;
        element_size = _element_size;
        direction = _direction;
    }

    char* get_ptr() {return pointer;};
    int get_element_size() {return element_size;};
    TraversalDirection get_direction() {return direction;};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

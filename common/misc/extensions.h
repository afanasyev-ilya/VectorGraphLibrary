#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool has_extension(string const &_full_string, string const &_extension)
{
    if (_full_string.length() >= _extension.length())
    {
        return (0 == _full_string.compare (_full_string.length() - _extension.length(), _extension.length(), _extension));
    }
    else
    {
        return false;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void remove_extension(string &_full_string)
{
    size_t lastdot = _full_string.find_last_of(".");
    if (lastdot == std::string::npos)
        return;
    _full_string = _full_string.substr(0, lastdot);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void add_extension(string &_full_string, string const &_extension)
{
    if(has_extension(_full_string, _extension))
        remove_extension(_full_string);
    _full_string += _extension;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

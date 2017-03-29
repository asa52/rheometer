#include <boost/python.hpp>
#include <Python.h>

int x = 100; 

int main()
{
    return 0;
}

BOOST_PYTHON_MODULE(example)
{
  boost::python::scope().attr("x") = x; // example.x
}
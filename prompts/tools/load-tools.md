# `load_tools`

Load complete definitions for selected tools from the supplied tool catalog.
Pass their exact names in `names`. The result lists loaded names and may include
their descriptions and argument schemas. Complete definitions join the API tool
list and system definitions on the next model turn. Read those supplied
definitions before choosing arguments.

Loaded tools remain available for this conversation. Repeated loads are safe
and supply the definitions on subsequent requests. Loading does not execute the selected tools
or grant access to tools excluded by the session's restrictions.

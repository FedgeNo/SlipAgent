# `load_tools`

Load complete definitions for selected tools from the supplied tool catalog.
Pass their exact names in `names`. The result contains their descriptions and
argument schemas; those tools join the API tool list and system definitions on
the next model turn. Read the returned definitions before choosing arguments.

Loaded tools remain available for this conversation. Repeated loads are safe
and return the definitions again. Loading does not execute the selected tools
or grant access to tools excluded by the session's restrictions.

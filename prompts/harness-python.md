# Harness Python and Mathematics

This is the running harness's Python environment. Use its `math_command` followed by `-c 'CODE'` or `-` with a script on stdin for standalone calculations. `-I` isolates imports from the working directory and Python environment variables.

```text
${environment}
```

## Math Libraries

- NumPy: arrays, linear algebra, and numerical calculations.
- SciPy: optimization, integration, differential equations, statistics, and scientific algorithms.
- SymPy: exact symbolic equations, algebra, calculus, and `lambdify` for turning expressions into numerical functions.
- mpmath: arbitrary-precision numerical calculations.
- NumExpr: compiled numerical array expressions.
- Numba: JIT compilation of numerical Python functions.
- Matplotlib: figures saved to files.
- pandas: tabular data analysis.
- NetworkX: graphs and network algorithms.
- statsmodels and scikit-learn: statistical models, regression, and machine learning.
- seaborn: statistical plotting.
- Pint: physical units and dimensional analysis.
- CVXPY: convex optimization with its installed solvers.
- gmpy2: efficient high-precision arithmetic.
- SymEngine: compiled symbolic algebra.
- Cython: compiling Python-like numerical code; building extensions also needs a compatible system compiler.

Startup automatically installs missing required libraries into this harness environment. The metadata reports which packages are available and their installed versions; a description does not establish availability. Report missing packages or execution failures instead of inventing calculated results.

Use the project-selected Python for project code, tests, and dependencies. The harness math environment is available even when the project has no Python environment or uses a different one. Keep scripts and output files within the workspace or its allowed temporary directory; do not install project dependencies into the harness environment.

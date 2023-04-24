# cgurobi.pxd

cdef extern from "gurobi_c.h":

    ctypedef struct GRBmodel:
        pass

    ctypedef struct GRBenv:
        pass

    ctypedef struct GRBsvec:
        int len
        int* ind
        double* val

    int __stdcall GRBloadenv(GRBenv **envP, const char *logfilename)

    int __stdcall GRBnewmodel(GRBenv *env, GRBmodel **modelP, const char *Pname, int numvars,
              double *obj, double *lb, double *ub, char *vtype,
              char **varnames)

    int __stdcall GRBsetdblattrelement(GRBmodel *model, const char *attrname,
                       int element, double newvalue)

    int __stdcall GRBsetintattr(GRBmodel *model, const char *attrname, int newvalue)

    int __stdcall GRBaddconstr(GRBmodel *model, int numnz, int *cind, double *cval,
               char sense, double rhs, const char *constrname);

    int __stdcall GRBaddconstrs(GRBmodel *model, int numconstrs, int numnz,
                int *cbeg, int *cind, double *cval,
                char *sense, double *rhs, char **constrnames);

    int __stdcall GRBaddvar(GRBmodel *model, int numnz, int *vind, double *vval,
                    double obj, double lb, double ub, char vtype,
                    const char *varname)

    int __stdcall GRBaddvars(GRBmodel *model, int numvars, int numnz,
                     int *vbeg, int *vind, double *vval,
                     double *obj, double *lb, double *ub, char *vtype,
                     char **varnames)

    int __stdcall GRBoptimize(GRBmodel *model)

    int __stdcall GRBBinvRowi(GRBmodel *model, int i, GRBsvec *x)

    int __stdcall GRBgetBasisHead(GRBmodel *model, int *bhead)

    int __stdcall GRBgetdblattrelement(GRBmodel *model, const char *attrname,
                           int element, double *valueP)
    
    int __stdcall GRBsetdblattrelement(GRBmodel *model, const char *attrname,
                           int element, double newvalue)

    int __stdcall GRBgetdblattr(GRBmodel *model, const char *attrname, double *valueP)
    
    int __stdcall GRBsetdblattr(GRBmodel *model, const char *attrname, double newvalue)

    int __stdcall GRBgetintattrarray(GRBmodel *model, const char *attrname,
                         int first, int len, int *values)

    int __stdcall GRBgetintattr(GRBmodel *model, const char *attrname, int *valueP)

    int __stdcall GRBgetdblattrarray(GRBmodel *model, const char *attrname,
                     int first, int len, double *values)

    int __stdcall GRBsetdblattrarray(GRBmodel *model, const char *attrname,
                     int first, int len, double *newvalues)

    int __stdcall GRBsetparam(GRBenv *env, const char *paramname, const char *value)

    int __stdcall GRBsetintparam(GRBenv *env, const char *paramname, int value)

    int __stdcall GRBsetdblparam(GRBenv *env, const char *paramname, double value)

    int __stdcall GRBsetstrparam(GRBenv *env, const char *paramname, const char *value)

    int __stdcall GRBreadmodel(GRBenv *env, const char *filename, GRBmodel **modelP)

    int __stdcall GRBread(GRBmodel *model, const char *filename)

    int __stdcall GRBwrite(GRBmodel *model, const char *filename)

    int GRB_INFINITY

    # Constraint senses

    char GRB_LESS_EQUAL    
    char GRB_GREATER_EQUAL 
    char GRB_EQUAL         

    # Variable types

    char GRB_CONTINUOUS 
    char GRB_BINARY     
    char GRB_INTEGER    
    char GRB_SEMICONT   
    char GRB_SEMIINT    

    # Objective sense

    int GRB_MINIMIZE
    int GRB_MAXIMIZE

    # variable attributes related to the current solution
    char GRB_DBL_ATTR_X

    # model solution attributes
    char GRB_DBL_ATTR_OBJVAL

    # model status
    int GRB_INT_ATTR_STATUS
    int GRB_OPTIMAL

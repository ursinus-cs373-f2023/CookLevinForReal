"""
Programmer: Chris Tralie
Purpose: To provide a basic CNF class, along with a vanilla
implementation of a Davis-Putnam-Logemann-Loveland (DPLL) SAT solver
"""

class CNF:
    def __init__(self):
        self.clauses = []
        self.vars = set([])
        
    def __repr__(self):
        return " ^ ".join(["(" + " V ".join([["¬",""][int(val)]+var for (var, val) in clause]) + ")" for clause in self.clauses])

    def add_clause(self, clause):
        """
        Add an OR clause to the CNF, indicating whether
        each variable has a NOT in front of it

        Parameters
        ----------
        clause: list of (hashable, bool)
            (variable name, [False/True])
        """
        self.clauses.append([(c[0], c[1]) for c in clause]) # Clone the clause
        self.vars = self.vars.union(set(c[0] for c in clause))

    def init_random_cnf(self, n_literals, n_clauses, seed=0):
        """
        Reset this to a random CNF formula

        Parameters
        ----------
        n_literals: int
            Number of literals in the formula
        n_clauses: int
            Number of clauses in the formula
        seed: int
            Seed to use when generating the formula
        """
        import numpy as np
        np.random.seed(seed)
        self.clauses = []
        self.vars = set([])
        for _ in range(n_clauses):
            clause = []
            for _ in range(1+np.random.randint(n_literals)):
                var = ["x{}".format(np.random.randint(n_literals))]
                var.append([False, True][np.random.randint(2)])
                clause.append(var)
            self.add_clause(clause)
    
    def is_satisfied(self, cert):
        """
        Check to see if a particular certificate satisfies this 
        CNF formula

        Parameters
        ----------
        cert: dictionary
            key: variable name, value: [False/True] 
        
        Returns
        -------
        True if this certificate satisfies the CNF
        False otherwise
        """
        is_satisfied = True
        i = 0
        while is_satisfied and i < len(self.clauses):
            clause = self.clauses[i]
            at_least_one = False
            for (var, val) in clause:
                at_least_one = at_least_one or (val == cert[var])
            is_satisfied = is_satisfied and at_least_one
            i += 1
        return is_satisfied

    def _solve_brute(self, idx, cert, vars):
        """
        Helper method for brute force solver

        Parameters
        ----------
        idx: int
            Index of variable currently being processed
        cert: dictionary
            key: variable name, value: [False/True] 
        vars: list of hashable
            List of variables in the order they are checked
        """
        res = False
        if idx == len(self.vars):
            res = self.is_satisfied(cert)
        else:
            # Try assigning False to this variable and continue
            cert[vars[idx]] = False
            res = self._solve_brute(idx+1, cert, vars)
            if not res:
                # Try assigning True to this variable and continue
                cert[vars[idx]] = True
                res = self._solve_brute(idx+1, cert, vars)
        return res
    
    def solve_brute(self):
        """
        Do a brute force 2^N solver for N literals

        Returns
        -------
        cert: dictionary
            key: variable name, value: [False/True] 
            Or an empty dictionary if it is not satisfiable
        """
        idx = 0
        cert = {}
        vars = sorted(list(self.vars))
        if not self._solve_brute(idx, cert, vars):
            cert = {}
        return cert

    def _filter_clauses(self, cnf, var, val):
        """
        Helper method for DPLL
        Remove every clause that contains this literal with this value, and
        delete this literal from every clause where it shows up as the complement

        Parameters
        ----------
        cnf: CNF
            A cnf object holding the currently simplified formula
        var: hashable
            Variable
        val: bool
            Value to assign to this variable
        """
        filtered_clauses = []
        for j in range(len(cnf.clauses)):
            filtered_clause = []
            to_keep = True
            for (varj, valj) in cnf.clauses[j]:
                if var == varj:
                    if val == valj:
                        # This clause is true so no need to keep it
                        to_keep = False
                else:
                    # All other variables are retained
                    filtered_clause.append((varj, valj))
            if to_keep:
                if len(filtered_clause) == 0:
                    # There are no variables left that we could satisfy,
                    # so we have an unsatisfiable clause
                    return False 
                else:
                    filtered_clauses.append(filtered_clause)
        cnf.clauses = filtered_clauses
        return True

    def _solve_dpll(self, cnf, cert):
        """
        Recursive helper method for DPLL solver

        Parameters
        ----------
        cnf: CNF
            A cnf object holding the currently simplified formula
        cert: dictionary
            key: variable name, value: [False/True] 
        """
        ## Step 1: Find and remove unit clauses
        i = 0
        while i < len(cnf.clauses):
            clause = cnf.clauses[i]
            if len(clause) == 1:
                # Assign the appropriate value to this literal
                (var, val) = clause[0]
                cert[var] = val
                if not self._filter_clauses(cnf, var, val):
                    return False
                i = 0 # Go back to beginning and start search again
            else:
                i += 1
        
        ## Step 2: Find pure literals and remove them from expressions
        literals = {}
        for clause in cnf.clauses:
            for (var, val) in clause:
                if not var in literals:
                    literals[var] = set([])
                literals[var].add(val)
        pure_literals = {}
        for var, vals in literals.items():
            if len(vals) == 1:
                val = list(vals)[0]
                pure_literals[var] = val
                cert[var] = val
                if not self._filter_clauses(cnf, var, val):
                    return False

        ## Step 3: Continue the recursion if necessary
        if len(cnf.clauses) == 0:
            return True # All clauses have been satisfied
        else:
            # Choose a literal that's left and try adding it and its negation
            clauses_before = [[(c[0], c[1]) for c in clause] for clause in cnf.clauses]
            cert_before = {k:v for k, v in cert.items()}
            for literal in cnf.vars.difference(set(cert.keys())):
                for val in [False, True]:
                    cnf.clauses = [[(c[0], c[1]) for c in clause] for clause in clauses_before]
                    cnf.clauses.append([(literal, val)])
                    cert.clear()
                    for k, v in cert_before.items():
                        cert[k] = v
                    cert[literal] = val
                    if self._solve_dpll(cnf, cert):
                        return True
        return False

    def solve(self):
        """
        Find a solution to the CNF formula if it exists, using
        the DPLL algorithm

        Returns
        -------
        cert: dictionary
            key: variable name, value: [False/True] 
            Or an empty dictionary if it is not satisfiable
        """
        # Make a new temporary CNF to store state
        cnf = CNF()
        for clause in self.clauses:
            cnf.add_clause(clause)
        cert = {}
        if self._solve_dpll(cnf, cert):
            # Add "True" arbitrarily for any variables that are unassigned
            for literal in cnf.vars.difference(set(cert.keys())):
                cert[literal] = True
        else:
            cert = {}
        return cert
from cnf import CNF
import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, V, seed):
        """
        Initialize a random graph

        Parameters
        ----------
        V: int
            Number of vertices
        seed: int
            Random seed (each seed will give a different graph)
        """
        np.random.seed(seed)
        self.V = V
        self.edges = set([])
        ## Step 1: Generate a random permutation, and
        ## add all of the edges
        perm = np.random.permutation(V)
        for i in range(V):
            # Take out pair of indices in permutation
            i1 = perm[i]
            i2 = perm[(i+1)%V]
            # Add this edge (convention lower index goes first)
            i1, i2 = min(i1, i2), max(i1, i2)
            self.edges.add((i1, i2))


        ## Step 2: Add some additional edges to make the 
        ## problem harder
        # Figure out what edges are left
        edges_left = []
        for i in range(V):
            for j in range(i+1, V):
                if (i, j) not in self.edges:
                    edges_left.append((i, j))
        # Pick some edges in edges_left to add
        # Want to add enough edges to make it a little hard
        # to find the cycle, but too many so that it's trivial
        E = int(V**(5/4))
        E = min(E, V*(V-1)//2)
        print("{:.3f} %% edges".format(100*E/(V*(V-1)/2)))
        for i in np.random.permutation(len(edges_left))[0:E-V]:
            self.edges.add(edges_left[i])

    def draw(self, perm=[]):
        """
        perm: list of V indices
            Permutation certificate of the ham cycle
        """
        V = self.V
        theta = np.linspace(0, 2*np.pi, V+1)[0:V]
        x = np.cos(theta)
        y = np.sin(theta)
        plt.scatter(x, y, s=100, zorder=100)
        for i in range(V):
            plt.text(x[i]+0.05, y[i], "{}".format(i))
        ## Draw each edge
        for (i, j) in self.edges:
            plt.plot([x[i], x[j]], [y[i], y[j]], c='k', linewidth=4)
        ## Draw the certificate permutation
        for k in range(len(perm)):
            i = perm[k]
            j = perm[(k+1)%V]
            plt.plot([x[i], x[j]], [y[i], y[j]], c='C1', linestyle='--')

    def get_at_most_one(self, cnf):
        """
        Fill in the constraint clauses that enforce
        at most one 1 in rows and columns

        Parameters
        ----------
        cnf: CNF
            CNF formula we're building
        """
        V = self.V
        for i in range(V):
            for j in range(V):
                for k in range(V):
                    # Each row has at most one 1
                    if k != j:
                        cnf.add_clause([ ( (i, j), False ), ( (i, k), False ) ])
                    # Each column has at most one 1
                    if k != i:
                        cnf.add_clause([ ( (i, j), False ), ( (k, j), False ) ])

    def get_at_least_one(self, cnf):
        """
        Fill in the constraints that enforce at least one 1 in each row

        Parameters
        ----------
        cnf: CNF
            CNF formula we're building
        """
        V = self.V
        for i in range(V):
            clause = []
            for j in range(V):
                clause.append( ( (i, j), True ) )
            cnf.add_clause(clause)
    
    def enforce_edges(self, cnf):
        """
        Fill in the constraints that make sure we're not adding edges
        that are not in the graph

        Parameters
        ----------
        cnf: CNF
            CNF formula we're building
        """
        V = self.V
        not_edges = set([])
        for i in range(V):
            for j in range(i+1, V):
                if not (i, j) in self.edges:
                    not_edges.add((i, j))
                    not_edges.add((j, i))
        for k in range(V):
            for (i, j) in not_edges:
                cnf.add_clause([( (k, i), False ), ( ((k+1)%V, j), False) ])

    def get_cnf_formula(self):
        """
        Do a reduction from this problem to SAT by filling in 
        CNF formulas

        Returns
        -------
        CNF: CNF Formula corresponding to the reduction
        """
        cnf = CNF()
        self.get_at_most_one(cnf)
        self.get_at_least_one(cnf)
        self.enforce_edges(cnf)
        return cnf
    
    def solve(self):
        cnf = self.get_cnf_formula()
        cert = cnf.solve_glucose()
        # Translate SAT solution back to my language
        perm = [0]*self.V
        for (i, j), val in cert.items():
            if val:
                perm[i] = j
        return perm

    def check_cert(self, perm):
        """
        Check a certificate in my language

        i.e. Make sure that perm is actually a valid
        hamiltonian cycle
        """
        is_valid = True
        ## Step 1: Check that it's a permutation
        if len(np.unique(perm)) != len(perm):
            is_valid = False
        ## Step 2: Check that they're edges in the graph
        for k in range(len(perm)):
            i = perm[k]
            j = perm[(k+1)%len(perm)]
            is_valid = is_valid and ( ((i, j) in self.edges or (j, i) in self.edges) )
        return is_valid
        
g = Graph(40, 0)
perm = g.solve()
print(g.check_cert(perm))
g.draw(perm)
plt.show()
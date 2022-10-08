"""
Solution stub for the River Problem.

Fill in the implementation of the `River_problem` class to match the
representation that you specified in problem XYZ.
"""
from searchProblem import Search_problem, Arc


class River_problem(Search_problem):
    def start_node(self):
        """returns start node"""
        start = ('F', 'H', 'Fx', 'G')
        return start

    def is_goal(self, node):
        """is True if node is a goal"""
        if node == ('Goal',):
            return True
        else:
            return False

    def neighbors(self, node):
        """returns a list of the arcs for the neighbors of node"""
        if node == ('F', 'H', 'Fx', 'G'):
            neigh = ('Fx', 'G')
            return [Arc(node, neigh, 1)]

        if node == ('Fx', 'G'):
            neigh = ('F', 'Fx', 'G')
            return [Arc(node, neigh, 1)]

        if node == ('F', 'Fx', 'G'):
            neigh1 = ('G',)
            neigh2 = ('Fx',)
            return [Arc(node, neigh1, 1), Arc(node, neigh2, 1)]

        if node == ('G',):
            neigh = ('F', 'H', 'G')
            return [Arc(node, neigh, 1)]

        if node == ('F', 'H', 'G'):
            neigh = ('H',)
            return [Arc(node, neigh, 1)]

        if node == ('Fx',):
            neigh = ('F', 'H', 'Fx')
            return [Arc(node, neigh, 1)]

        if node == ('F', 'H', 'Fx'):
            neigh = ('H',)
            return [Arc(node, neigh, 1)]

        if node == ('H',):
            neigh = ('F', 'H')
            return [Arc(node, neigh, 1)]

        if node == ('F', 'H'):
            neigh = ('Goal',)
            return [Arc(node, neigh, 1)]

    def heuristic(self, n):
        """Gives the heuristic value of node n."""

        return 1


def main():
    print("F = Farmer\nH = Hen\nFx = Fox\nG = Grain")
    print("Nodes indicate the object(s) on the LEFT side of the river")


if __name__ == '__main__':
    main()

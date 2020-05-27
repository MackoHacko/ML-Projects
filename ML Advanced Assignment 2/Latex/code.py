class Node:
    
    def __init__(self, name, cat):
        

        self.cat         = []
        for c in cat:
            self.cat.append(c)
        self.ancestor    = None
        self.descendants = []
        self.sample      = None
        self.smaller     = np.zeros(len(cat[0]))
        self.name        = name


def makeSmaller(root):

''' Generates s(u,x_i) for each node 
and stores them in self.smaller '''
    if root:
        for child in root.descendants:
            makeSmaller(child)
        if root.descendants == []:
            root.smaller[int(root.sample)] = 1
        else:
            for i in range(len(root.cat[0])):
                temp1 = 1.0
                for child in root.descendants: 
                    temp2 = 0.0
                    for j in range(len(child.cat[0])):
                        temp2 += child.cat[i][j]*child.smaller[j]
                    temp1 *= temp2
                root.smaller[i] = temp1
                                    
def treeLikelihood(root):

''' Calls makeSmaller to prepare 
tree through evaluating each s(u,x_i), 
then returns the tree likelihood '''
    makeSmaller(root)
    likelihood = 0.0
    for i in range(len(root.cat[0])):
        likelihood += root.smaller[i]*root.cat[0][i]
    return likelihood
import numpy as np
import pickle


class Node:
    
    def __init__(self, name, cat):
        
        self.name = name
        self.cat     = []
        for c in cat:
            self.cat.append(c)
        self.ancestor    = None
        self.descendants = []
        self.sample      = None
    
    
    
def append_3_to_list(tmp_list, a, b, c):
    tmp_list.append(a)
    tmp_list.append(b)
    tmp_list.append(c)
    
    
def convert_to_list(array):
    out = []
    n = len(array)
    m = array[0].shape[0]
    for i in range(n):
        temp = []
        for j in range(m):
            temp.append(array[i][j])
        out.append(temp)
    return out
    
    
def  dirichlet(a):
    return np.random.dirichlet(a)
    


class Tree:
    
    def __init__(self):
        
        self.root   = None
        self.layers = 0
        
        
    def create_random_tree(self, k, max_depth = 5, max_branching = 5, alpha = []):
        """
            This methods creates a tree. The name of each node is simply its position
            when traversing in a breadth-first manner.
            
            k = the number of categorical variables
            max_depth   = How deep can the tree potentially be?
            max_branching = How many children can each node have?
            alpha = the hyperparameters of the dirichlet distribution used to sample theta
            
        
        """
        if len(alpha) == 0:
            alpha = [1.0]*k # no preferences 
        if len(alpha) != k or np.sum(np.array(alpha) < 0) != 0:
            print("Alpha needs to contain k positive values! ")
            return None
        node_count  = 1
        depth_count = 1
        self.root  = Node(str(node_count), [dirichlet(alpha)])
        curr_layer = [self.root]
        # while we have not reached max_depth or there are no leaves left
        # grow the tree
        init = 1
        while curr_layer != [] and depth_count < max_depth: 
            next_layer = []
            for elem in curr_layer:
                # draw how many children the current node will have
                num_children = np.random.randint(init,max_branching + 1) 
                init = 0
                for child in range(num_children):
                    node_count = node_count + 1
                    # draw a categorical distribution for each value the parent could have
                    cat = []
                    for theta in range(k):
                        cat.append(dirichlet(alpha))
                    child_node = Node(str(node_count), cat)
                    child_node.ancestor = elem
                    elem.descendants.append(child_node)
                    next_layer.append(child_node)   
            curr_layer  = next_layer
            depth_count = depth_count + 1
             
        return self.convert_tree_to_format()
        
        
        
    def load_params(self, params):
        """
            The input should be given in Newick form:
                [A,B,[C,D]E]F;
                
                   ________A 
                  |
                F |________B
                  |          ________C
                  |         |
                  |________E
                            |________D
                            
            But we are working in python. So we will work with lists that look
            the following way:
                
            Tree  = ['F', pF, ['A', pA, [], 'B', pB, [], 'E', pE, ['C', pC, [], 'D', pD, []]]]
                
            Each variable has a name (in string format), a list of categorical parameters,
            e.g. pF = [0.3, 0.2, 0.5], and a list of child nodes.
        """
        
        self.root = Node(params[0], params[1])
        self.__add_children(self.root, params[2])

    def __add_children(self, node, params):
        """
            Adds the children and their children recursively to "node".
        
        """
        if len(params)  == 0:
            return
        num_children = len(params) 
        for child in range(0,num_children,3):
            child_node = Node(params[child], params[child+1])
            child_node.ancestor = node
            node.descendants.append(child_node)
            self.__add_children(child_node, params[child + 2])
            
    def print_tree(self, print_sample = False):
        """
            Prints tree layer by layer without correct spacing for children.
            print_sample (bool) determines whether we also print the current
            sample. 
        """
        curr_layer = [self.root]
        while curr_layer != []:
            string = ''
            next_layer = []
            for elem in curr_layer:
                string = string + elem.name  + ' ' 
                if (print_sample and elem.sample != None):
                    string = string[:-1] + ':' + str(elem.sample)  + ' ' 
                for child in elem.descendants:
                    next_layer.append(child)   
            print(string)
            curr_layer = next_layer

    def convert_tree_to_format(self):
        """
            Returns a sample in our python / Newick format. 
        """
        if self.root  == None:
            print("No tree yet! ")
            return
        return self.__add_to_format(self.root,[])
    
     
        
    def __add_to_format(self, node, out_list ):
        """
            Recursive function that adds each descendant and its value.
            Keep in mind that lists are passed by reference, not by value.
        """
        node_chi = []
        append_3_to_list(out_list,node.name, convert_to_list(node.cat), node_chi)
        for child in node.descendants:
            self.__add_to_format(child, node_chi )
        return out_list
        
     
            
            
     
        
         

        
        
            
            
 
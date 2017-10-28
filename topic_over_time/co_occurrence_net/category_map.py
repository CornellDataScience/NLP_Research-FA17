class CategoryNode(object):
    def __init__(self, category):
        # string for the category name
        self.c = category
        # counting the appearance of the categories
        self.counter = 1
        # set of strings -> frequency
        self.cooccurrence = {}

    def add_coocurence(self, clist):
        '''
        Add a list of items and if it does not equal to self, add the item to the
        cooccurrence dictionary

        Input:
            clist(list) : list of items that co-occured with this node
        '''
        for c in clist:
            if c != self.c:
                if c in self.cooccurrence.keys():
                    self.cooccurrence[c] += 1
                else:
                    self.cooccurrence[c] = 1

class CategoryMap(object):
    def __init__(self):
        # string -> CategoryNode
        self.categories = {}

    def observe(self, clist):
        '''
        If the item is not registered in the network, instantiate the Node object
        Else, increment the counter and for each node, update the coocurence

        Input:
            clist(list) : list of items in one sequence
        '''
        for c in clist:
            # first appearance
            if c not in self.categories.keys():
                self.categories[c] = CategoryNode(c)
            # already in the system
            else:
                node = self.categories[c]
                node.counter += 1
            self.categories[c].add_coocurence(clist)

    def display_top_n(self, n):
        '''
        print n nodes in a decreasing order of appearance

        Input:
            n(int) : number of items to display
        '''
        categories = {}
        for c in self.categories.keys():
            categories[c] = self.categories[c].counter

        ordered_c = sorted(categories.items(), key=lambda x: x[1], reverse = True)
        print (ordered_c[:n])

    def get_subcategories(self, c):
        '''
        return the list of subcategories of c
        We define subcategories of c as a list of tags that have been co-occurred with specified tag with fewer total number of appearance that the specified tags

        For example,
        if the tag 'Chinese' has appeared with 'Sushi Bar' and 'Restaurants'
        and total number of 'Sushi Bar' appearance is fewer than 'Chinese' then
        'Sushi Bar' is a sub category of 'Chinese'

        Input:
            c(str) : name of item
        Output:
            ordered_sub_c(list) : list of subcategories and its frequency
        '''
        if c not in self.categories.keys():
            raise Exception('category not found in the system')
        else:
            c_node = self.categories[c]
            # get all the cooccurrence
            ordered_sub_c = sorted(c_node.cooccurrence.items(), key=lambda x: x[1], reverse = True)
            # filter out larger categories
            ordered_sub_c = [sc for sc in ordered_sub_c if self.categories[sc[0]].counter < c_node.counter ]
            return ordered_sub_c

    def shared_categories(self, c1, c2, sub = False):
        '''
        Return the list of shared categories between c1 and c2.
        if sub = True, then only returns subcategories

        Input:
            c1(str) : name of item1
            c2(str) : name of item2
            (Optional)
            sub(bol) : if True, only returns subcategories
        Output:
            list of subcategories
        '''
        if c1 not in self.categories.keys():
            raise Exception('category not found in the system')
        else:
            c_node = self.categories[c1]
            # get all the cooccurrence
            if sub:
                ordered_sub_c1 = self.get_subcategories(c1)
            else:
                ordered_sub_c1 = sorted(c_node.cooccurrence.items(), key=lambda x: x[1], reverse = True)
            ordered_sub_c1 = [c1[0] for c1 in ordered_sub_c1]

        if c2 not in self.categories.keys():
            raise Exception('category not found in the system')
        else:
            c_node = self.categories[c2]
            # get all the cooccurrence
            if sub:
                ordered_sub_c2 = self.get_subcategories(c2)
            else:
                ordered_sub_c2 = sorted(c_node.cooccurrence.items(), key=lambda x: x[1], reverse = True)
            ordered_sub_c2 = [c2[0] for c2 in ordered_sub_c2]

        return set(ordered_sub_c1) & set(ordered_sub_c2)

    def build_graph(self, seq):
        '''
        Build a graph from the list of sequence

        Input:
            clist(list list) : list of sequecnes
        '''
        for item in seq:
            self.observe(item)

if __name__ == '__main__':
    # from utils import *
    #
    # DATAPASS = '../../../yelp10/dataset/business.json'
    # business = load_json_to_df(DATAPASS)

    G = CategoryMap()
    G.build_graph(business.categories)

    print (G.get_subcategories('Chinese'))

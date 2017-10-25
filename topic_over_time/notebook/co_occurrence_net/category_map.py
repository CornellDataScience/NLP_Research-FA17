class CategoryNode(object):
    def __init__(self, category):
        # string for the category name
        self.c = category
        # counting the appearance of the categories
        self.counter = 1
        # set of strings -> frequency
        self.cooccurrence = {}

    def add_coocurence(self, clist):
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

    def observed(self, clist):
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
        categories = {}
        for c in self.categories.keys():
            categories[c] = self.categories[c].counter

        ordered_c = sorted(categories.items(), key=lambda x: x[1], reverse = True)
        print (ordered_c[:n])

    def get_subcategories(self, c):
        '''
        return the list of subcategories of c
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

if __name__ == '__main__':
    from utils import *

    DATAPASS = '../../../yelp10/dataset/business.json'
    business = load_json_to_df(DATAPASS)

    G = CategoryMap()

    for c in business.categories:
        G.observed(c)

    print (G.get_subcategories('Chinese'))

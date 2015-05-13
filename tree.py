__author__ = 'zephyrYin'

class Node:
    def __init__(self, value = -1, label = '', children = {}):
        self.attribute = value
        self.label = label
        self.children = children
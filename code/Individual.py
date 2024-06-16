import config as cfg


class Individual:
    def __init__(self, id, data=None, generation=None):  ## temp fix data=1
        self.id = id
        self.data = data
        self.generation = generation
        self.parents = []
        self.children = []
        self.score = None
        self.sample_id = None

    def __str__(self):
        return f"Individual {self.data}"

    def __repr__(self):
        return f"Individual {self.data}"

    def add_child(self, child):
        if len(self.children) >= cfg.NUMBER_OF_CHILDREN:
            raise Exception(f"Too many children for individual {self}")
        if len(child.parents) >= cfg.NUMBER_OF_PARENTS:
            raise Exception(f"Too many parents for individual {child}")
        self.children.append(child)
        child.parents.append(self)

    def add_parent(self, parent):
        if len(self.parents) >= cfg.NUMBER_OF_PARENTS:
            raise Exception(f"Too many parents for individual {self}")
        if len(parent.children) >= cfg.NUMBER_OF_CHILDREN:
            raise Exception(f"Too many children for individual {parent}")
        self.parents.append(parent)
        parent.children.append(self)

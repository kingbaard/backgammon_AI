class Player:
    def __init__(self, id):
        self.id = id
        self.bar = 0
        if id == 0:
            self.bar_loc = -1
        else:
            self.bar_loc = 24
        self.goal = 0
        self.can_bear = False

    def __str__(self):
        return f"Player {self.id + 1}"
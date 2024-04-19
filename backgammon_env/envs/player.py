class Player:
    def __init__(self, id, bar_i, type):
        self.id = id
        self.bar_i = bar_i
        self.goal = 0
        self.can_bear = False
        self.type = type

    def __str__(self):
        return f"Player {self.id + 1}"
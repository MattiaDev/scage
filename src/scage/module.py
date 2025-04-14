import random


class Module:
    def __init__(self, start_symbol, min_expansions, max_expansions):
        self.start_symbol = start_symbol
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions
        self.layers = []

    def initialise(self, grammar, reuse, initial_expansions: list):
        num_expansions = random.choice(initial_expansions)

        # Initialise layers
        for idx in range(num_expansions):
            if idx > 0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                self.layers.append(self.layers[r_idx])
            else:
                self.layers.append(grammar.initialise(self.start_symbol))

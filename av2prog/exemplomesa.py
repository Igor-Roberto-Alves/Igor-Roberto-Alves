import matplotlib.pyplot as plt
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa import Agent
from mesa.time import RandomActivation
import random
#iashduiashdasidhsai
class TreeCell(Agent):
    """
    A tree cell agent that can be in one of three states:
    "Fine", "On Fire", or "Burned Out".
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.condition = "Fine"

    def step(self):
        """
        If the tree is "On Fire", spread the fire to neighboring "Fine" trees.
        """
        if self.condition == "On Fire":
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            for neighbor in neighbors:
                if isinstance(neighbor, TreeCell) and neighbor.condition == "Fine":
                    neighbor.condition = "On Fire"
        
            self.condition = "Burned Out"

class ForestFire(Model):
    """
    Simple Forest Fire model.
    """
    def __init__(self, width=100, height=100, density=0.65):
        """
        Create a new forest fire model.
        Args:
            width, height: The size of the grid to model
            density: What fraction of grid cells have a tree in them.
        """
        self.current_id = 0  # Initialize current_id for unique agent IDs
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            {
                "Fine": lambda m: self.count_trees(m, "Fine"),
                "On Fire": lambda m: self.count_trees(m, "On Fire"),
                "Burned Out": lambda m: self.count_trees(m, "Burned Out"),
            }
        )

        # Place a tree in each cell with probability = density
        for x in range(width):
            for y in range(height):
                if self.random.random() < density:
                    tree = TreeCell(self.next_id(), self)
                    self.grid.place_agent(tree, (x, y))
                    self.schedule.add(tree)
                    # Set all trees in the first column on fire.
                    if x == 0:
                        tree.condition = "On Fire"

        self.running = True
        self.datacollector.collect(self)

    def next_id(self):
        """
        Generate the next unique ID for an agent.
        """
        self.current_id += 1
        return self.current_id

    def step(self):
        """
        Advance the model by one step.
        """
        self.schedule.step()
        self.datacollector.collect(self)
        
        # Halt if no more trees are on fire
        if self.count_trees(self, "On Fire") == 0:
            self.running = False

    def count_trees(self, model, condition):
        """
        Helper method to count trees in a given condition in the model.
        """
        return sum(1 for agent in model.schedule.agents if agent.condition == condition)

# Run the model
fire = ForestFire(width=100, height=100, density=0.6)
print(fire.grid)

while fire.running:
    fire.step()

# Collect data for visualization
results = fire.datacollector.get_model_vars_dataframe()

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(results["Fine"], label="Fine")
plt.plot(results["On Fire"], label="On Fire")
plt.plot(results["Burned Out"], label="Burned Out")
plt.xlabel("Step")
plt.ylabel("Number of Trees")
plt.title("Forest Fire Simulation")
plt.legend()
plt.show()



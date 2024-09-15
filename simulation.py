import numpy as np
import quads


class Simulation:
    def __init__(self, container, **kwargs):
        self.container = container
        self.plants = []
        self.states = []
        self.r_max_global = kwargs.get('r_max_global', None)

    def add_plant(self, plant):
        if self.container.is_within_bounds(plant.pos):
            self.plants.append(plant)
            self.container.insert(quads.Point(
                plant.pos[0], plant.pos[1], data=plant))
            return True
        return False

    def site_quality(self, pos):
        bb_half_width = self.r_max_global
        max_plant_area = np.pi*self.r_max_global**2
        bb = self.container.create_bb(pos, bb_half_width)
        plants_nearby = [point.data for point in self.container.query(bb)]
        if len(plants_nearby) == 0:
            return 0
        density_nearby = sum(
            plant.r**2*np.pi for plant in plants_nearby)/(len(plants_nearby)*max_plant_area)
        return density_nearby

    def step(self):
        for plant in self.plants:
            if plant.is_dead:
                self.plants.remove(plant)
                # REMOVE FROM CONTAINER AS WELL
                continue

            collisions = self.container.get_collisions(
                plant, self.r_max_global)
            plant.resolve_collisions(collisions)
            plant.update(self)

        self.container.update(self.plants)
        self.states.append(self.container.copy())

    def export_states(self):
        simulation_states_dicts = [
            [plant.__dict__ for plant in state] for state in self.states]
        return simulation_states_dicts

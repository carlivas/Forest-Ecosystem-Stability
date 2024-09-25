import numpy as np
import quadT

from plant import Plant


class Simulation:
    def __init__(self, **kwargs):
        half_width = kwargs.get('half_width', 0.5)
        half_height = kwargs.get('half_height', half_width)
        qt_capacity = kwargs.get('qt_capacity', 4)

        self.land_quality = kwargs.get('land_quality', None)
        self.density_check_range = kwargs.get('density_check_range', None)

        self.qt = quadT.QuadTree(
            (0, 0), half_width, half_height, capacity=qt_capacity)
        self.qt_new = self.qt

    def add_plant(self, plant):
        self.qt_new.insert(quadT.Point(plant.pos, data=plant))

    def site_quality(self, pos):
        bb_half_width = self.density_check_range

        bb = quadT.BoundingCircle(pos, bb_half_width)
        plants_nearby = [point.data for point in self.qt.query(bb)]
        if len(plants_nearby) == 0:
            return 0
        density_nearby = sum(
            plant.A for plant in plants_nearby)/bb.area
        return density_nearby + self.land_quality

    def step(self):
        self.qt_new = quadT.QuadTree(center=self.qt.boundary.center, half_width=self.qt.boundary.half_width,
                                     half_height=self.qt.boundary.half_height, capacity=self.qt.capacity)

        plants = [point.data for point in self.qt.all_points()]

        # First Phase: Update all plants
        for plant in plants:
            plant.update(self)

        # Second Phase: Collect non-dead plants and add them to the new state
        for plant in plants:
            if not plant.is_dead:
                self.add_plant(plant.copy())
                del plant

        self.qt = self.qt_new

    def get_state(self):
        return self.qt


def initialize_random(**kwargs):
    simulation = Simulation(**kwargs)
    half_width = kwargs.get('half_width', 0.5)
    half_height = kwargs.get('half_height', half_width)
    num_plants = kwargs.get('num_plants', None)
    i = 0
    while i < num_plants:
        rand_pos = np.random.uniform(-half_width,
                                     half_width, 2)
        this_plant_kwargs = kwargs.copy()
        this_plant_kwargs['id'] = i
        this_plant_kwargs['r'] = np.random.uniform(
            kwargs.get('r_min'), kwargs.get('r_max'))
        plant = Plant(rand_pos, **this_plant_kwargs)
        simulation.add_plant(plant)
        print(f'Planted {i+1}/{num_plants} plants', end='\r')
        i += 1

    return simulation

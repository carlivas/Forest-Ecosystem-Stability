import numpy as np
import quadT


class Plant:
    def __init__(self, pos: np.ndarray, **kwargs):
        self.pos = pos

        self.kwargs = kwargs

        self.r_start = kwargs.get('r_start', 0)
        self.r_max = kwargs.get('r_max', 0.05)
        self.r = kwargs.get('r', self.r_start)
        self.d = 2*self.r
        self.A = np.pi*self.r**2

        self.growth_rate = kwargs.get('growth_rate', 0.0001)
        self.age_max = (self.r_max - self.r_start)/self.growth_rate

        self.reproduction_chance = kwargs.get('reproduction_chance', 0.001)
        self.reproduction_range = kwargs.get(
            'reproduction_range', self.r)

        self.young_color = (69, 194, 51)
        self.old_color = (163, 194, 122)
        self.color = kwargs.get('color', self.young_color)
        self.is_dead = kwargs.get('is_dead', False)
        self.is_colliding = kwargs.get('is_colliding', False)

    def __eq__(self, other):
        return np.all(self.pos == other.pos) and self.r == other.r

    def set_color(self, color):
        self.color = color

    def color_from_size(self):
        t = np.clip(self.r / self.r_max, 0, 1)
        return tuple(np.array(self.young_color) * (1 - t) + np.array(self.old_color) * t)

    def grow(self):
        self.r = self.r + self.growth_rate

    def mortality(self):
        # Check if the plant has reached its maximum size
        if self.r > self.r_max:
            self.die()

    def copy(self):
        return Plant(**self.__dict__)

    def die(self):
        self.is_dead = True

    def reproduce(self, simulation):
        new_pos = self.pos + np.random.uniform(-self.reproduction_range,
                                               self.reproduction_range, 2)

        # Determine if reproduction is successful based on chance and site quality
        p = self.reproduction_chance * simulation.site_quality(new_pos)
        if np.random.rand() < p:
            # Create a new plant and add it to the simulation
            new_plant_kwargs = self.kwargs.copy()
            new_plant_kwargs['r_start'] = self.r_start
            new_plant_kwargs['r'] = self.r_start
            new_plant = Plant(new_pos, **new_plant_kwargs)
            simulation.add_plant(new_plant)

    def compete(self, other):
        if self.r != other.r:
            (self if self.r < other.r else other).die()
        else:
            (self if np.random.rand() > 0.5 else other).die()

    def compete(self, other):
        if np.random.rand() > self.r / (self.r + other.r):
            self.die()
        else:
            other.die()

    def resolve_collisions(self, collisions):
        if len(collisions) == 0:
            self.is_colliding = False
        else:
            self.is_colliding = True
            for other in collisions:
                other.is_colliding = True
                self.compete(other)

    def update(self, simulation, collisions):
        self.resolve_collisions(collisions)
        self.mortality()
        self.grow()
        self.reproduce(simulation)
        self.set_color(self.color_from_size())
        return

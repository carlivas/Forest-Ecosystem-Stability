import numpy as np
import copy


class Plant:
    def __init__(self, pos: np.ndarray, r, id=None, **kwargs):
        self.pos = pos
        self.id = id

        self.kwargs = kwargs

        self.r_min = kwargs['r_min']
        self.r_max = kwargs['r_max']
        self.r = r
        self.d = 2*self.r
        self.area = np.pi*self.r**2

        self.growth_rate = kwargs['growth_rate']
        self.age_max = (self.r_max - self.r_min)/self.growth_rate

        self.species_germination_chance = kwargs['species_germination_chance']
        self.dispersal_range = kwargs['dispersal_range']

        self.is_dead = kwargs.get('is_dead', False)
        self.is_colliding = kwargs.get('is_colliding', False)
        self.generation = kwargs.get('generation', 0)

    def grow(self):
        self.r = self.r + self.growth_rate
        self.d = 2*self.r
        self.area = np.pi*self.r**2

    def mortality(self):
        if self.r > self.r_max:
            self.die()

    def copy(self):
        return Plant(**self.__dict__)

    def die(self):
        self.is_dead = True

    def reproduce(self, simulation):
        if self.species_germination_chance > 0:
            if self.r > self.r_min * 100 and not self.is_dead:
                # Determine if reproduction is successful based on chance and site quality
                reproduction_chance = simulation.quality_nearby(
                    self.pos) * self.species_germination_chance

                if reproduction_chance > np.random.rand():
                    rand_ang = np.random.rand() * 2 * np.pi
                    new_dir = np.array([np.cos(rand_ang), np.sin(rand_ang)])
                    d = np.random.normal(self.r, self.dispersal_range)
                    new_pos = self.pos + new_dir * d

                    new_plant_kwargs = self.kwargs.copy()
                    new_plant_kwargs['r_min'] = self.r_min
                    new_plant_kwargs['r'] = self.r_min
                    new_plant_kwargs['is_colliding'] = False
                    new_plant_kwargs['is_dead'] = False
                    new_plant_kwargs['generation'] = self.generation + 1

                    simulation.add(Plant(new_pos, **new_plant_kwargs))

    def compete(self, other_plant):
        p = 0.5
        # p = np.random.rand()
        if p > self.r / (self.r + other_plant.r):
            self.die()
        else:
            other_plant.die()

    def resolve_collisions(self, collisions):
        for other_plant in collisions:
            self.compete(other_plant)

    def update(self, simulation):
        self.grow()

        collisions = simulation.get_collisions(self)
        self.resolve_collisions(collisions)

        self.reproduce(simulation)
        self.mortality()
        return

    def copy(self):
        return Plant(**self.__dict__)

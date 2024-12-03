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
        if self.species_germination_chance > 0 and not self.is_dead:
            # Determine if reproduction is successful based on chance and site quality
            # new_pos = self.pos + np.random.uniform(-self.dispersal_range,
            #                                         self.dispersal_range, size=2)
            new_pos = self.pos + np.random.normal(
                0, self.dispersal_range, size=2)

            if simulation.pos_in_box(new_pos):

                # reproduction_chance = simulation.local_density(
                #     new_pos) * self.species_germination_chance
                reproduction_chance = (simulation.local_density(
                    new_pos) + simulation.land_quality) * self.species_germination_chance

                # if reproduction_chance > np.random.uniform(0, 1 - simulation.land_quality):
                if reproduction_chance > np.random.uniform(0, 1):

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

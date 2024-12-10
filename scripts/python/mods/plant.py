import numpy as np
import scipy as sp
import copy


class Plant:
    def __init__(self, pos: np.ndarray, r, id=None, **kwargs):
        self.pos = pos
        self.id = id

        self.kwargs = kwargs

        self.r = r
        self.r_min = kwargs['r_min']
        self.r_max = kwargs['r_max']
        self.growth_rate = kwargs['growth_rate']

        too_big = self.r > self.r_max
        too_small = self.r < self.r_min
        if too_big or too_small:
            r = self.r
            r_min = self.r_min
            r_max = self.r_max
            # print(f'Plant: {r=} is outside of the bounds [{r_min=}, {r_max=}]')

        self.d = 2*self.r
        self.area = np.pi*self.r**2

        self.age_max = (self.r_max - self.r_min)/self.growth_rate

        self.species_germination_chance = 1
        # self.dispersal_range = kwargs['dispersal_range']

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

    # def disperse(self, sim):
    #     if self.species_germination_chance > 0 and not self.is_dead:
    #         new_pos = self.pos + sp.stats.cauchy.rvs(
    #             loc=0, scale=self.dispersal_range, size=2)

    #         dispersal_chance = sim.local_density(
    #             new_pos) * sim.precipitation_func(sim.t) * self.species_germination_chance

    #         if dispersal_chance > np.random.uniform(0, 1 - sim.land_quality):

    #             new_plant_kwargs = self.kwargs.copy()
    #             new_plant_kwargs['r_min'] = self.r_min
    #             new_plant_kwargs['r'] = self.r_min
    #             new_plant_kwargs['is_colliding'] = False
    #             new_plant_kwargs['is_dead'] = False
    #             new_plant_kwargs['generation'] = self.generation + 1

    #             sim.add(Plant(new_pos, **new_plant_kwargs))

    # def disperse_old(self, sim):
    #     if self.species_germination_chance > 0 and not self.is_dead:
    #         new_pos = self.pos + np.random.normal(
    #             0, self.dispersal_range, size=2)

    #         dispersal_chance = (sim.local_density(
    #             new_pos) + sim.land_quality) * self.species_germination_chance

    #         if dispersal_chance > np.random.uniform(0, 1):

    #             new_plant_kwargs = self.kwargs.copy()
    #             new_plant_kwargs['r_min'] = self.r_min
    #             new_plant_kwargs['r'] = self.r_min
    #             new_plant_kwargs['is_colliding'] = False
    #             new_plant_kwargs['is_dead'] = False
    #             new_plant_kwargs['generation'] = self.generation + 1

    #             sim.add(Plant(new_pos, **new_plant_kwargs))

    def compete(self, other_plant):
        if self.r < other_plant.r:
            self.die()

        elif self.r > other_plant.r:
            other_plant.die()

        elif self.r == other_plant.r:
            if np.random.rand() > 0.5:
                self.die()
            else:
                other_plant.die()

    def resolve_collisions(self, collisions):
        for other_plant in collisions:
            self.compete(other_plant)

    def update(self, sim):
        self.grow()

        collisions = sim.get_collisions(self)
        self.resolve_collisions(collisions)

        # self.disperse(sim)
        self.mortality()
        return

    def copy(self):
        return Plant(**self.__dict__)

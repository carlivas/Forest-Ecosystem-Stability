import numpy as np
import scipy as sp
import copy


class Plant:
    def __init__(self, pos: np.ndarray, r, r_min=0.1, r_max=30, growth_rate=0.1, dispersal_range=90, id=None, **kwargs):
        self.pos = pos
        self.r = r

        self.id = id

        self.d = 2*self.r
        self.area = np.pi*self.r**2

        self.r_min = r_min
        self.r_max = r_max
        self.growth_rate = growth_rate

        self.species_germination_chance = 1
        self.dispersal_range = dispersal_range

        self.age_max = (self.r_max - self.r_min)/self.growth_rate

        # self.is_dead = kwargs.get('is_dead', False)
        # self.is_colliding = kwargs.get('is_colliding', False)
        # self.generation = kwargs.get('generation', 0)

        self.is_dead = False
        self.is_colliding = False

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

    def disperse(self, sim):
        if self.species_germination_chance > 0 and not self.is_dead:
            # angle = np.random.uniform(0, 2*np.pi)
            # distance = np.random.lognormal(
            #     mean=0, sigma=1.5) * self.dispersal_range
            # new_pos = self.pos + \
            #     np.array([np.cos(angle), np.sin(angle)]) * distance

            new_pos = self.pos + np.random.normal(
                0, self.dispersal_range, size=2)
            dispersal_chance = sim.local_density(
                new_pos) * sim.precipitation(sim.t) * self.species_germination_chance

            if dispersal_chance <= 0:
                return
            elif dispersal_chance > np.random.uniform(0, 1 - sim.land_quality):
                sim.add(Plant(new_pos, r=self.r_min, r_min=self.r_min, r_max=self.r_max,
                        growth_rate=self.growth_rate, dispersal_range=self.dispersal_range))

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
        if not sim.spinning_up:
            self.disperse(sim)
        self.mortality()
        return

    def copy(self):
        return Plant(**self.__dict__)

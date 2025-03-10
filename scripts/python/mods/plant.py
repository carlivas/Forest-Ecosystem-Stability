import numpy as np
import scipy as sp
import copy


class Plant:
    def __init__(self, id, x, y, r, r_min, r_max, growth_rate, dispersal_range, density_range, maturity_size, germination_chance=1, n_offspring=1, species_id = -1, **kwargs):
        self.id = id
        self.x = x
        self.y = y
        self.r = r

        self.d = 2*self.r
        self.area = np.pi*self.r**2

        self.r_min = r_min
        self.r_max = r_max
        self.growth_rate = growth_rate

        self.maturity_size = maturity_size
        self.germination_chance = germination_chance
        self.dispersal_range = dispersal_range
        self.density_range = density_range
        self.n_offspring = n_offspring
        
        self.species_id = species_id

        self.is_dead = False
        self.is_colliding = False

    def grow(self):
        self.r = self.r + self.growth_rate
        self.d = 2*self.r
        self.area = np.pi*self.r**2

    def mortality(self):
        if self.r > self.r_max:
            self.die()

    def die(self):
        self.is_dead = True

    # Maybe move this to a simulation class
    def disperse(self, sim):
        if self.r < self.maturity_size:
            return
        
        n = self.n_offspring * sim.time_step
        decimal_part = n % 1
        if np.random.rand() < decimal_part:
            n = int(n) + 1
        else:
            n = int(n)

        # new_plants = []
        new_positions = np.full((n, 2), np.nan)
        for i in range(n):
            if self.germination_chance > 0 and not self.is_dead:
                new_pos = np.array([self.x, self.y]) + \
                    np.random.normal(0, self.dispersal_range, 2)
                new_positions[i] = new_pos
                
        sim.attempt_germination(new_positions, self)
                    
                
        #         # DO BOUNDARY CONDITIONS HERE

        #         dispersal_chance = max(sim.land_quality, sim.local_density(
        #             new_pos) * sim.precipitation * self.germination_chance)

        #         if dispersal_chance > np.random.uniform(0, 1):
        #             new_plants.append(
        #                 Plant(
        #                     id=sim.id_generator.get_next_id(),
        #                     x=new_pos[0],
        #                     y=new_pos[1],
        #                     r=self.r_min,
        #                     r_min=self.r_min,
        #                     r_max=self.r_max,
        #                     growth_rate=self.growth_rate,
        #                     germination_chance=self.germination_chance,
        #                     dispersal_range=self.dispersal_range,
        #                     density_range=self.density_range,
        #                     species_id=self.species_id,
        #                     maturity_size=self.maturity_size,
        #                 )
        #             )
        # sim.add(new_plants)

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

    def update(self, sim):
        self.grow()
        self.disperse(sim)
        self.mortality()
        return

    def copy(self):
        return Plant(**self.__dict__)
    
    def pos(self):
        return np.array([self.x, self.y])


class PlantSpecies:
    def __init__(self, species_id=-1, r_min=1, r_max=30, growth_rate = 0.1, dispersal_range=90, density_range=90, maturity_size=1, germination_chance = 1, n_offspring=1,  **kwargs):
        self.species_id = species_id
        self.name = kwargs.get('name', f'Species {species_id}')
        self.r_min = r_min
        self.r_max = r_max
        self.growth_rate = growth_rate
        self.dispersal_range = dispersal_range
        self.density_range = density_range
        self.maturity_size = maturity_size
        self.germination_chance = germination_chance
        self.n_offspring = n_offspring

    def create_plant(self, id, x, y, r, **kwargs):
        return Plant(id=id, x=x, y=y, r=r, **self.__dict__)
    
    def copy(self):
        return PlantSpecies(**self.__dict__)
        
        
    
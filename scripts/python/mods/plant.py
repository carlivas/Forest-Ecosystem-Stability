import numpy as np
import scipy as sp
import copy
from itertools import compress


class Plant:
    def __init__(self, id, x, y, r, r_min, r_max, growth_rate, dispersal_range, density_range, maturity_size, germination_chance=1, n_offspring=1, species_id = -1, **kwargs):
        self.id = id
        self.x = x
        self.y = y
        self.r = r

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

    def mortality(self):
        if self.r > self.r_max:
            self.is_dead = True

    # Maybe move this to a simulation class
    def disperse(self, sim):
        if self.r < self.maturity_size:
            return np.empty((0,2), dtype=float)
        
        n = self.n_offspring * sim.time_step
        decimal_part = n % 1
        if np.random.rand() < decimal_part:
            n = int(n) + 1
        else:
            n = int(n)

        new_positions = np.empty((n,2), dtype=float)
        # COULD BE PARALLELIZED
        for i in range(n):
            if self.germination_chance > 0 and not self.is_dead:
                new_pos = np.array([self.x, self.y]) + \
                    np.random.normal(0, self.dispersal_range, 2)
                new_positions[i] = new_pos
                
        return new_positions

    def compete(self, other_plant):
        if self.r < other_plant.r:
            self.is_dead = True

        elif self.r > other_plant.r:
            other_plant.is_dead = True

        elif self.r == other_plant.r:
            if np.random.rand() > 0.5:
                self.is_dead = True
            else:
                other_plant.is_dead = True

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
        
class PlantCollection:
    def __init__(self, plants=None):
        self.plants = []
        self.positions = np.empty((0, 2), dtype=float)
        self.radii = np.empty(0, dtype=float)
        self.is_dead = np.empty(0, dtype=bool)
        
        if plants:
            for plant in plants:
                self.add_plant(plant)
            self.update()
    
    def grow(self):
        # COULD BE PARALLELIZED
        for i, plant in enumerate(self.plants):
            plant.grow()
            self.radii[i] = plant.r
                
    def disperse(self, sim):
        dispersed_positions = np.empty((0, 2))
        parent_species = []
        # COULD BE PARALLELIZED
        for plant in self.plants:
            pos = plant.disperse(sim)
            if len(pos) > 0:
                dispersed_positions = np.vstack((dispersed_positions, pos))
                parent_species.extend([PlantSpecies(**plant.__dict__)] * len(pos))
        return dispersed_positions, parent_species
    
    def mortality(self):
        # COULD BE PARALLELIZED
        for i, plant in enumerate(self.plants):
            plant.mortality()
            self.is_dead[i] = plant.is_dead
        
    def add_plant(self, plant):
        self.plants.append(plant)
        self.positions = np.vstack((self.positions, plant.pos()))
        self.radii = np.append(self.radii, plant.r)
        self.is_dead = np.append(self.is_dead, plant.is_dead)

    def append(self, plant):
        self.add_plant(plant)
        
    def add_plants(self, plants):
        for plant in plants:
            self.add_plant(plant)

    def remove_dead_plants(self):
        alive_indices = np.where(~self.is_dead)[0]
        self.plants = list(compress(self.plants, ~self.is_dead))
        self.positions = self.positions[alive_indices]
        self.radii = self.radii[alive_indices]
        self.is_dead = self.is_dead[alive_indices]
            
    def update_values(self):
        # COULD BE PARALLELIZED
        for i, plant in enumerate(self.plants):
            self.radii[i] = plant.r
            self.is_dead[i] = plant.is_dead
    
    def __getitem__(self, index):
        return self.plants[index]
    
    def __len__(self):
        return len(self.plants)
    
    def __iter__(self):
        return iter(self.plants)
    
    def __str__(self):
        return f'PlantCollection with {len(self.plants)} plants'
import numpy as np
import scipy as sp
import copy


class Plant:
    def __init__(self, pos: np.ndarray, r: float, r_min: float, r_max: float, growth_rate: float, dispersal_range: float, _m: float):
        self.pos = pos
        self._m = _m
        self.r = r * _m
        self.r_min = r_min * _m
        self.r_max = r_max * _m
        self.growth_rate = growth_rate * _m
        self.dispersal_range = dispersal_range * _m

        self.species_germination_chance = 1
        self.d = 2*self.r
        self.area = np.pi*self.r**2

        self.is_dead = False
        self.is_colliding = False

    def grow(self, dt):
        self.r = self.r + self.growth_rate * dt
        self.d = 2*self.r
        self.area = np.pi*self.r**2

    def mortality(self):
        if self.r > self.r_max:
            self.die()

    def copy(self):
        
        return Plant(
            pos=self.pos,
            r=self.r_min,
            r_min=self.r_min,
            r_max=self.r_max,
            growth_rate=self.growth_rate,
            dispersal_range=self.dispersal_range,
            _m=self._m
        )

    def die(self):
        self.is_dead = True

# Maybe move this to a simulation class
    def disperse(self, sim):
        n = sim.spawn_rate * sim.time_step
        decimal_part = n % 1
        if np.random.rand() < decimal_part:
            n = int(n) + 1
        else:
            n = int(n)

        new_plants = []
        for _ in range(n):
            if self.species_germination_chance > 0 and not self.is_dead:
                new_pos = self.pos + \
                    np.random.normal(0, self.dispersal_range, 2)

                dispersal_chance = max(sim.land_quality, sim.local_density(
                    new_pos) * sim.precipitation(sim.t) * self.species_germination_chance)

                if dispersal_chance > np.random.uniform(0, 1):
                    new_plants.append(
                        Plant
                        (
                            new_pos,
                            r=self.r_min,
                            r_min=self.r_min,
                            r_max=self.r_max,
                            growth_rate=self.growth_rate,
                            dispersal_range=self.dispersal_range,
                            _m=self._m
                        )
                    )
        sim.add(new_plants)

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

# Maybe move this to a simulation class
    def update(self, sim):
        self.grow(sim.time_step)

        collisions = sim.get_collisions(self)
        self.resolve_collisions(collisions)
        if not sim.spinning_up:
            self.disperse(sim)
        self.mortality()
        return

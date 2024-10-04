import numpy as np
# import quadT


class Plant:
    def __init__(self, pos: np.ndarray, **kwargs):
        self.pos = pos

        self.kwargs = kwargs

        self.r_min = kwargs.get('r_min')
        self.r_max = kwargs.get('r_max')
        self.r = kwargs.get('r', self.r_min)
        self.d = 2*self.r
        self.area = np.pi*self.r**2

        self.growth_rate = kwargs.get('growth_rate')
        self.age_max = (self.r_max - self.r_min)/self.growth_rate

        self.reproduction_chance = kwargs.get('reproduction_chance')
        self.reproduction_range = kwargs.get(
            'reproduction_range')

        self.young_color = (69, 194, 51)
        self.old_color = (163, 194, 122)
        self.color = kwargs.get('color', self.young_color)
        self.is_dead = kwargs.get('is_dead', False)
        self.is_colliding = kwargs.get('is_colliding', False)
        self.generation = kwargs.get('generation', 0)

    def set_color(self, color):
        self.color = color

    def color_from_size(self):
        t = np.clip(self.r / self.r_max, 0, 1)
        return tuple(np.array(self.young_color) * (1 - t) + np.array(self.old_color) * t)

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
        rand_ang = np.random.rand() * 2 * np.pi
        new_dir = np.array([np.cos(rand_ang), np.sin(rand_ang)])
        d = np.random.uniform(self.r, self.reproduction_range)
        new_pos = self.pos + new_dir * d

        # Determine if reproduction is successful based on chance and site quality
        p = simulation.site_quality(new_pos) * self.reproduction_chance

        # if self.reproduction_thresholds[0] < p and p < self.reproduction_thresholds[1]:
        if p > np.random.rand():
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

    # def get_collisions(self, simulation):
    #     self.is_colliding = False
    #     collisions = []
    #     indices = simulation.kt.query_ball_point(
    #         x=self.pos, r=self.d, workers=-1)
    #     for i in indices:
    #         other_plant = simulation.plants[i]
    #         if other_plant != self:
    #             if check_collision(self, other_plant):
    #                 self.is_colliding = True
    #                 other_plant.is_colliding = True
    #                 collisions.append(other_plant)
    #     return collisions

    def resolve_collisions(self, collisions):
        for other_plant in collisions:
            self.compete(other_plant)

    def update(self, simulation):
        self.grow()

        collisions = simulation.get_collisions(self)
        self.resolve_collisions(collisions)

        self.reproduce(simulation)
        self.mortality()
        self.set_color(self.color_from_size())
        return

    def copy(self):
        return Plant(**self.__dict__)

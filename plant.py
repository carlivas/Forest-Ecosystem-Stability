import numpy as np
import quadT


def check_collision(p1, p2):
    return np.sum((p1.pos - p2.pos) ** 2) < (p1.r + p2.r) ** 2


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
            'reproduction_range', 10*self.r)

        self.young_color = (69, 194, 51)
        self.old_color = (163, 194, 122)
        self.color = kwargs.get('color', self.young_color)
        self.is_dead = kwargs.get('is_dead', False)
        self.is_colliding = kwargs.get('is_colliding', False)
        self.generation = kwargs.get('generation', 0)

        self.id = kwargs.get('id', None)

    def __str__(self):
        if self.id is not None:
            return f"Plant {self.id}"
        else:
            return f"Plant at {self.pos:.5f}"

    def set_color(self, color):
        self.color = color

    def color_from_size(self):
        t = np.clip(self.r / self.r_max, 0, 1)
        return tuple(np.array(self.young_color) * (1 - t) + np.array(self.old_color) * t)

    def grow(self):
        self.r = self.r + self.growth_rate
        self.d = 2*self.r
        self.A = np.pi*self.r**2

    def mortality(self):
        if self.r > self.r_max:
            self.die()

    # def mortality(self):
    #     p = np.exp((self.r/self.r_max)**2) - 1
    #     p /= np.exp(1) - 1
    #     if np.random.rand() < p:
    #         self.die()

    def copy(self):
        return Plant(**self.__dict__)

    def die(self):
        self.is_dead = True

    def reproduce(self, simulation):
        rand_ang = np.random.rand() * 2 * np.pi
        new_dir = np.array([np.cos(rand_ang), np.sin(rand_ang)])
        d = np.random.uniform(2*self.r, self.reproduction_range)
        new_pos = self.pos + new_dir * d

        # Determine if reproduction is successful based on chance and site quality
        p = self.reproduction_chance * \
            simulation.site_quality(new_pos)
        if np.random.rand() < p:

            new_plant_kwargs = self.kwargs.copy()
            new_plant_kwargs['r_start'] = self.r_start
            new_plant_kwargs['r'] = self.r_start
            new_plant_kwargs['is_colliding'] = False
            new_plant_kwargs['is_dead'] = False
            new_plant_kwargs['generation'] = self.generation + 1

            simulation.add_plant(Plant(new_pos, **new_plant_kwargs))

    def compete(self, other):
        # p = 0.5
        p = np.random.rand()
        if p > self.r / (self.r + other.r):
            self.die()
        else:
            other.die()

    def get_collisions(self, simulation):
        self.is_colliding = False
        collisions = []
        bb = quadT.BoundingCircle(self.pos, self.d)
        for point in simulation.qt.query(bb):
            other_plant = point.data
            if other_plant != self:
                if check_collision(self, other_plant):
                    self.is_colliding = True
                    other_plant.is_colliding = True
                    collisions.append(other_plant)
        return collisions

    # def resolve_collisions(self, collisions):
    #     for other in collisions:
    #         self.compete(other)
    #     self.is_colliding = False

    def resolve_collisions(self, simulation):
        collisions = self.get_collisions(simulation)
        for other in collisions:
            self.compete(other)

    # def update(self, simulation, collisions):
    def update(self, simulation):
        # self.resolve_collisions(collisions)
        self.resolve_collisions(simulation)
        self.mortality()
        self.grow()
        self.reproduce(simulation)
        self.set_color(self.color_from_size())
        return

    def copy(self):
        return Plant(**self.__dict__)

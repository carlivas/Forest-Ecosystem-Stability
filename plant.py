import numpy as np
import quads


class Plant:
    def __init__(self, pos: np.ndarray, **kwargs):
        self.pos = pos
        self.r_start = kwargs.get('r_start', 0)
        self.r_max = kwargs.get('r_max', 0.05)
        self.r = kwargs.get('r', self.r_start)

        self.growth_rate = kwargs.get('growth_rate', 0.0001)
        self.age_max = (self.r_max - self.r_start)/self.growth_rate

        self.reproduction_chance = kwargs.get('reproduction_chance', 0.001)
        self.reproduction_attempts = kwargs.get('reproduction_attempts', 1)
        self.reproduction_range = kwargs.get(
            'reproduction_range', 10*self.r_max)

        self.young_color = (0.27, 0.76, 0.2)
        self.old_color = (0.64, 0.76, 0.44)
        self.color = kwargs.get('color', self.young_color)
        self.is_dead = kwargs.get('is_dead', False)

        self.kwargs = kwargs

    def set_color(self, color):
        self.color = color

    def color_from_size(self):
        t = np.clip(self.r / self.r_max, 0, 1)
        return tuple(np.array(self.young_color) * (1 - t) + np.array(self.old_color) * t)

    def grow(self):
        self.r += self.growth_rate

    def mortality(self):
        # Check if the plant has reached its maximum size
        if self.r > self.r_max:
            self.die()

    # def mortality(self):
    #     # Check if the plant has reached its maximum age
    #     if np.random.rand() > 1 - (1 - 4/self.age_max)**self.age_max:
    #         self.die()

    def copy(self):
        return Plant(**self.__dict__)

    def die(self):
        self.is_dead = True

    def reproduce(self, simulation):
        # Attempt reproduction a specified number of times
        for i in range(self.reproduction_attempts):
            # Generate a random angle and distance for the new plant's position
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, self.reproduction_range)

            if distance < 2*self.r_max:
                continue

            new_pos = self.pos + \
                np.array([np.cos(angle), np.sin(angle)]) * distance

            # Calculate the site quality at the new position
            q = simulation.site_quality(tuple(new_pos))

            # Determine if reproduction is successful based on chance and site quality
            if np.random.rand() < self.reproduction_chance * q:
                # Create a new plant and add it to the simulation
                new_plant = Plant(new_pos, **self.kwargs)
                simulation.add_plant(new_plant)

    def compete(self, other):
        if self.r < other.r:
            self.die()
        # else:
        #     other.die()

    # def compete(self, other):
    #     if np.random.rand() > self.r / (self.r + other.r):
    #         self.die()
    #     else:
    #         other.die()

    def resolve_collisions(self, collisions):
        for other in collisions:
            if other != self:
                self.compete(other)
                # continue

    def update(self, simulation):
        self.mortality()
        self.reproduce(simulation)
        self.grow()
        self.set_color(self.color_from_size())

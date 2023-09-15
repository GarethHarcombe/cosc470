import numpy as np

class Circle:
    def __init__(self, a, b, r, top_half) -> None:
        """
        circle in form:
          x = a + r cos(t)
          y = b + r sin(t)

        top_half: bool whether the circle is the top half or bottom half of the circle/track

        length of half circle = pi * r = 114.67m
        """
        self.a = a
        self.b = b
        self.r = r

        self.fx = lambda t: self.a + self.r * np.cos(t)
        self.fy = lambda t: self.b + self.r * np.sin(t)
        
        # derive y/x from above equations
        self.dydx = lambda t: -np.cos(t) / (np.sin(t))
        self.top_half = top_half


    def parametric_point(self, t):
        return (self.fx(t), self.fy(t))


    def project(self, x1, y1):
        """
        Given a point, find the closest projection onto the circle
        Return None if below/above the semi circle defined by self.top_half
        """
        t = np.arctan2(y1 - self.b, x1 - self.a)

        if self.top_half and (t > np.pi or t < 0):
            return None
        elif not self.top_half and t < np.pi and t > 0:
            return None
        else:
            return (self.fx(t), self.fy(t))


class Straight:
    def __init__(self, x, lower_bound, upper_bound, is_back_straight):
        """
        Straight: either back or home stretch of a track

        Attributes ~
            x: float - for a standard track, is either -36.5 (back straight) or 36.5 (home straight)
            lower_bound: float - southern y value where the straight ends (and goes into a bend)
            upper_bound: float - northern y value where the straight ends (and goes into a bend)
            is_back_straight: bool - True if the straight is a back straight. False if home straight
        """
        self.x = x
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_back_straight = is_back_straight


    def parametric_point(self, t, straight_length):
        if self.is_back_straight:
            return (self.x, self.upper_bound - (t / straight_length) * (self.upper_bound - self.lower_bound))
        else:
            return (self.x, self.lower_bound + (t / straight_length) * (self.upper_bound - self.lower_bound))


    def project(self, x1, y1):
        return (self.x, y1) if y1 < self.upper_bound and y1 > self.lower_bound else None


class Track:
    def __init__(self, r=36.5, s=84.39):
        """
        theoretical values for a track:
        https://www.dlgsc.wa.gov.au/sport-and-recreation/sports-dimensions-guide/athletics-track-events

        parametrically defined here: https://www.desmos.com/calculator/sikussppqz
        there are also tracks with shorter straights (around 80m) and 120m bends - these are ignored for now
        """

        self.s = s  # straight lengths
        self.r = r  # radius of the circles

        self.first_bend = Circle(0, s/2, r, top_half=True)
        self.back_straight = Straight(-r, -s/2, s/2, is_back_straight=True)
        self.last_bend = Circle(0, -s/2, r, top_half=False)
        self.home_straight = Straight( r, -s/2, s/2, is_back_straight=False)


    def parametric_point(self, t):
        """
        total t for one revolution is 2 * (t for each circle) + 2 * (t for each straight)
                                    = 2 * (pi) + 2 * (84.39 * pi / 114.67)
                                    = around 10.907... radians
        """
        straight_length = self.s * np.pi / (np.pi * self.r)    #  np.pi * np.pi * self.r / self.s 
        total_t = 2 * np.pi + 2 * straight_length
        t -= total_t * (t // total_t)   # normalise

        if t < np.pi:
            return self.first_bend.parametric_point(t)
        elif t < np.pi + straight_length:
            return self.back_straight.parametric_point(t - np.pi, straight_length)
        elif t < 2 * np.pi + straight_length:
            return self.last_bend.parametric_point(t - straight_length)
        else:
            return self.home_straight.parametric_point(t - (2 * np.pi + straight_length), straight_length)


    def project(self, x1, y1):
        """
        Projects any point onto the closest point on a 400m track. 
        Assumes 400m track is centered at (0, 0)
        """

        first_bend_point = self.first_bend.project(x1, y1)

        back_straight = self.back_straight.project(x1, y1)

        back_bend_point = self.last_bend.project(x1, y1)
        
        home_straight = self.home_straight.project(x1, y1)

        points = [first_bend_point, back_straight, back_bend_point, home_straight]
        filtered_p = [point for point in points if point is not None]

        return filtered_p[np.argmin([(point[0] - x1) ** 2 + (point[1] - y1) ** 2 for point in filtered_p])]


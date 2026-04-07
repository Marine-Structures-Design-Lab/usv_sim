import logging


# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- NORMALIZATION ----------
def normalized_direction_difference(theta1_deg, theta2_deg):
    """
    REQUIRES:
        - theta1_deg and theta2_deg are floats or ints representing angles in degrees.
        - Both angles are valid (i.e., any real number — wraparound is handled).

    MODIFIES:
        - Nothing.

    EFFECTS:
        - Converts both angles to 2D unit vectors on the unit circle.
        - Computes the Euclidean distance between the two vectors.
        - Normalizes the result to return a value in [0, 1].
        - Returns 0 if directions are the same, 1 if they are 180° apart.

    RETURNS:
        - float: normalized direction difference between theta1 and theta2, in [0, 1].
    """
def normalized_height_difference(h1, h2, max_height=6.0):
    """
    REQUIRES:
        - h1 and h2 are non-negative floats or ints representing significant wave heights (in meters).
        - max_height is a positive float representing the normalization factor (default is 6.0 meters).

    MODIFIES:
        - Nothing.

    EFFECTS:
        - Computes the absolute difference in wave height between h1 and h2.
        - Divides the result by max_height to normalize it to the range [0, 1].
        - Returns a float in [0, ∞), typically in [0, 1] if max_height is reasonable.

    RETURNS:
        - float: normalized wave height difference between h1 and h2.
    """
def normalized_period_difference(p1, p2, max_period=20.0):
    """
    REQUIRES:
        - p1 and p2 are positive floats or ints representing peak wave periods (in seconds).
        - max_period is a positive float representing the normalization factor (default is 20.0 seconds).

    MODIFIES:
        - Nothing.

    EFFECTS:
        - Computes the absolute difference in peak period between p1 and p2.
        - Divides the result by max_period to normalize it to the range [0, 1].
        - Returns a float in [0, ∞), typically in [0, 1] if max_period is reasonable.

    RETURNS:
        - float: normalized wave period difference between p1 and p2.
    """



def main():
    # given 2 points P1 and P2
    # calculate distance metric for each of the swells (3x3 matrix)
    # D(i, j) = a * abs(H_i - H_j) + b * abs(T_i - T_j) + c * dir_diff(theta_i - theta_j)
    """
            P2S1    P2S2    P2S3
    P1S1    x      x       x
    P1S2    x      x       x
    P1S3    x      x       x
    """
    # use Hungarian Algorithm (OR other matching algorithm to determine matching swells)
    # special case when one swell is changed from a level to the next...
    # return matching pairs ex. [(0, 0), (1, 2), (2, 1)]


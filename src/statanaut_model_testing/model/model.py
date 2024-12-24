from typing import Optional
from .rating import Rating
import numpy as np
from scipy.stats import norm


class Model:
    def _calculate_team_rating(
        self,
        ratings: list[Rating],
        alliance_name: Optional[str] = None,
    ) -> Rating:
        mu_summed = np.sum([rating.mu for rating in ratings])
        sigma_squared_summed = np.sum([(rating.sigma**2) for rating in ratings])

        return Rating(
            name=alliance_name,
            mu=np.float32(mu_summed),
            sigma=np.sqrt(sigma_squared_summed),
        )

    def predict_win(
        self,
        red_alliance: list[Rating],
        blue_alliance: list[Rating],
    ) -> tuple[float, float]:
        red_rating = self._calculate_team_rating(
            red_alliance,
            alliance_name="red",
        )
        blue_rating = self._calculate_team_rating(
            blue_alliance,
            alliance_name="blue",
        )

        red_prob = norm.cdf(
            x=(
                (red_rating.mu - blue_rating.mu)
                / np.sqrt((red_rating.sigma**2) + (blue_rating.sigma**2))
            ),
            loc=0,
            scale=1,
        )

        return (
            red_prob.item(),
            (1.0 - red_prob.item()),
        )

    def rate(
        self,
        red_alliance: list[Rating],
        blue_alliance: list[Rating],
        red_score: np.float32,
        blue_score: np.float32,
    ) -> tuple[list[Rating], list[Rating]]:

        red_score = np.float32(red_score)
        blue_score = np.float32(blue_score)

        red_win_probability, blue_win_probability = self.predict_win(
            red_alliance, blue_alliance
        )

        outcome = np.float32(1) if red_score > blue_score else np.float32(0)

        pass

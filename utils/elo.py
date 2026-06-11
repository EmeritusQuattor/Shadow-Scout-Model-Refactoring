class EloSystem:
    def __init__(self, k=32, initial_rating=1500):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}

    def get_rating(self, team_number):
        return self.ratings.get(str(team_number), self.initial_rating)

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, team_number, opponent_rating, score, opponent_score):
        rating = self.get_rating(team_number)
        expected = self.expected_score(score, opponent_score)

        actual = 1.0 if score > opponent_score else 0.0
        if score == opponent_score:
            actual = 0.5

        new_rating = rating + self.k * (actual - expected)
        self.ratings[str(team_number)] = new_rating
        return new_rating

    def update_match(self, red_team, blue_team, red_score, blue_score):
        for team in red_team:
            red_rating = self.get_rating(team)
            blue_rating = self.get_rating(blue_team[0])
            self.update(team, blue_rating, red_score, blue_score)

        for team in blue_team:
            blue_rating = self.get_rating(team)
            red_rating = self.get_rating(red_team[0])
            self.update(team, red_rating, blue_score, red_score)

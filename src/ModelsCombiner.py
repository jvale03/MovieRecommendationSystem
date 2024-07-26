
def calculate_scores(rec):
    scores = {}
    for i, movie in enumerate(rec):
        scores[movie] = 1/(i+1)
    return scores



def combine_results(content, collaborative):
    content_score = calculate_scores(content)
    collaborative_score = calculate_scores(collaborative)

    combined_scores = {}

    for movie, score in content_score.items():
        combined_scores[movie] = combined_scores.get(movie, 0) + (0.5 * score)

    for movie, score in collaborative_score.items():
        combined_scores[movie] = combined_scores.get(movie, 0) + (0.5 * score)

    sorted_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    sorted_movies = [movie for movie, _ in sorted_movies[:10]]

    return sorted_movies
PRODUCT_QUERY_PREFIX = """Given a product description and a review, answer whether the product would be suitable for kids. Answer with ONLY "Yes" or "No". Here are some examples first:
Example:
description: "1900 Classical Violin Music"
reviewText: "This is a comforting track that gives me peaceful time away from my children"
Answer: "No".

Example:
description: "Smooth funky jazz music!"
reviewText: "My son really loves listening to this!"
Answer: "Yes"

Here is the actual review:"""

SENTIMENT_QUERY_PREFIX = """Given the following ReviewText, answer whether the sentiment associated is "Positive" or "Negative". Answer with ONLY "Positive", "Negative", or "Neutral". Here are some examples first:
Example: 
ReviewText: "Very boring watch, the performances in this movie were horrible"
Answer: "Negative".

Example: 
ReviewText: "Entertaining fun time with family!"
Answer: "Positive"

Example: 
ReviewText: "Not sure how I feel about this movie..."
Answer: "Neutral"

Here is the actual review:
"""

MOVIE_RECOMMENDATION_QUERY_PREFIX = """
Given a set of movie descriptions and a critic review, use the critic review and the movie advertised by the description provided to recommend another movie. Answer with ONLY ONE movie title or "unsure". Here are some examples:
Example: 
Description = "A fun thrilling adventure movie!"
ReviewText = "Very boring watch, the performances in this movie were horrible"
Answer: Harry Potter

Example 2:
Description: "A factual and educational documentary of Napoleon Bonaparte"
ReviewText: "A dramatized telling of Napoleon's life"
Answer: Our Planet

Example 3:
Description: "A fun spin on Greek mythology through an adventure epic following a boy Percy Jackson"
ReviewText: "A fun fantasy adventure!"
Answer: Percy Jackson

Here is the actual description and review:
"""

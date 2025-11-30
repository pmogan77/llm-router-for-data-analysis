import copy
import json
import pandas as pd
import time

from ollama import chat

from prompt_templates import SENTIMENT_QUERY_PREFIX, MOVIE_RECOMMENDATION_QUERY_PREFIX

products_df = pd.read_csv('datasets/products.csv')
movies_df = pd.read_csv('datasets/movies.csv')
print('movies dataset size', len(movies_df))
print(movies_df.columns)

# Run the context-aware routing pattern
print("Initiating Context-Aware Routing Pattern...")

results = {
    'total_latency': [],
    'responses': [],
}

models = ['hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest', 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M']
results_per_model = {
    'latency': [],
    'responses': []
}
results_per_query = {
    model: copy.deepcopy(results_per_model) for model in models
}
results = {}

for model in models:
    for row in range(20):
        print(f'Row {row + 1}')
        movie_description = movies_df.loc[row, 'movieinfo']
        review_content = movies_df.loc[row, 'reviewcontent']

        print(movie_description)
        print(review_content)

        sentiment_query = f"""
        {SENTIMENT_QUERY_PREFIX}
        ReviewText: \"{review_content}\"
        Answer:
            """

        movie_recommendation_query = f"""
        {MOVIE_RECOMMENDATION_QUERY_PREFIX}
        Description: \"{movie_description}\"
        ReviewText: \"{review_content}\"
        Answer:
            """

        for query_type, query in [('sentiment', sentiment_query), ('recommendation', movie_recommendation_query)]:
            request_time = time.time()
            response = chat(model=model, messages=[
                {
                    'role': 'user',
                    'content': query,
                },
            ])
            response_time = time.time()
            if query_type not in results:
                results[query_type] = copy.deepcopy(results_per_query)
            results[query_type][model]['latency'].append(response_time - request_time)
            results[query_type][model]['responses'].append(response['message']['content'])

            print(response['message']['content'])

print(json.dumps(results, indent=4))
with open('output/single_model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

from f00889_prepare_data import *
model_name = 'google/tapas-base'
data = {'Actors': ['Brad Pitt', 'Leonardo Di Caprio', 'George Clooney'], 'Number of movies': ['87', '53', '69']}
queries = ['What is the name of the first actor?', 'How many movies has George Clooney played in?', 'What is the total number of movies?']
answer_coordinates = [[(0, 0)], [(2, 1)], [(0, 1), (1, 1), (2, 1)]]
answer_text = [['Brad Pitt'], ['69'], ['209']]

result = prepare_data(model_name, data, queries, answer_coordinates, answer_text)
print(result)

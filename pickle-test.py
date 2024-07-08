import pickle

some_data = list(range(100))

pickle.dump(some_data, open('pickle-test.pckl', 'wb'))
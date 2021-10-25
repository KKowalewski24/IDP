import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

min_value = 0.5
max_value = 2.5


def fitness(x):
    return (np.exp(x) * np.sin(10 * np.pi * x) + 1) / x + 5


def _decode(x, min_value, max_value, MAX):
    return min_value + x / MAX * (max_value - min_value)


def _mutate(x, number_of_bits):
    return x ^ (
        np.uint64(1) << np.random.randint(0, number_of_bits, dtype=np.uint64))


def _crossover(x, y, number_of_bits, MAX):
    crosspoint = np.random.randint(0, number_of_bits, dtype=np.uint64)
    right_mask = MAX >> crosspoint
    left_mask = ~right_mask
    return ((x & right_mask) | (y & left_mask),
            (x & left_mask) | (y & right_mask))


def _roulette_selection(population, selection_probabilities):
    return population[np.count_nonzero(
        selection_probabilities < np.random.rand())]


def sga(decimal_digits,
        number_of_populations,
        population_size,
        crossover_probability,
        mutation_probability,
        verbose=False):

    # find proper number of bits (required to be lower then 64 - the size of uint64)
    number_of_bits = np.ceil(
        np.log2((max_value - min_value) * (10**decimal_digits) + 1)).astype(
            np.uint8)
    if number_of_bits >= 64:
        raise ValueError(
            "Too high representation accuracy - number of subintervals (max value) can not be represented by uint64"
        )
    MAX = np.power(np.uint64(2), number_of_bits) - np.uint64(1)

    # create initial population
    population = np.random.randint(0,
                                   MAX + 1,
                                   size=(population_size, ),
                                   dtype=np.uint64)
    fitness_values = fitness(_decode(population, min_value, max_value, MAX))

    maxes = []
    for population_idx in range(number_of_populations):
        # calculate selection probabilities for population
        selection_probabilities = fitness_values / np.sum(fitness_values)
        for i in reversed(range(len(selection_probabilities))):
            selection_probabilities[i] = np.sum(selection_probabilities[:i +
                                                                        1])

        # create new population using genetic operators
        new_population = []
        while len(new_population) < population_size:
            r = np.random.rand()
            if r < mutation_probability:
                new_population.append(
                    _mutate(
                        _roulette_selection(population,
                                            selection_probabilities),
                        number_of_bits))
            elif r < mutation_probability + crossover_probability:
                new_population += _crossover(
                    _roulette_selection(population, selection_probabilities),
                    _roulette_selection(population, selection_probabilities),
                    number_of_bits, MAX)
            else:
                new_population.append(
                    _roulette_selection(population, selection_probabilities))
        population = np.array(new_population[:population_size])
        fitness_values = fitness(_decode(population, min_value, max_value,
                                         MAX))
        if verbose:
            maxes.append(np.max(fitness_values))
            print(f"{population_idx+1}/{number_of_populations}\t{maxes[-1]}")
    if verbose:
        plt.plot(maxes)
        plt.show()
    return _decode(population[np.argmax(fitness_values)], min_value, max_value,
                   MAX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("-M", type=int, required=True)
    parser.add_argument("-pc", type=float, required=True)
    parser.add_argument("-pm", type=float, required=True)
    parser.add_argument("-v", action='store_true')
    parser.add_argument("-R", type=int, required=False)
    args = parser.parse_args()

    if args.R is None:
        best_solution = sga(args.p, args.N, args.M, args.pc, args.pm, args.v)
        print(f"Best solution: {best_solution} -> {fitness(best_solution)}")
    else:
        counter = 0
        for i in tqdm(range(args.R)):
            best_solution = sga(args.p, args.N, args.M, args.pc, args.pm,
                                args.v)
            if fitness(best_solution) > 10.0:
                counter += 1
        print(f"Global optimum found {counter} times!")

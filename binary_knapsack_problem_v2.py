"""
An evolutionary algorithm for solving the Knapsack problem

In the binary knapsack problem, you are given a set of objects
    o_1, o_2, ..., o_k
along with their values
    v_1, v_2, ..., v_k
and their weights
    w_1, w_2, ..., w_k
The goal is to maximize the total value of the selected objects, subject to
a weight value constraint W.
"""

__author__ = 'Konstantinos Gkentsidis'
__license__ = 'BSDv3'

import numpy as np
import statistics


class Knapsack_problem:
    def __init__(self, numObjects):
        self.value = 2 ** np.random.randn(numObjects)
        self.weights = 2 ** np.random.randn(numObjects)
        self.capacity = 0.25 * np.sum(self.weights)
        self.numObjects = numObjects

        print("The array of values is: ", self.value)
        print("The array of weights is: ", self.weights)
        print("The total capacity is: ", self.capacity)
        print("The number of objects are: ", self.numObjects)

    def individual(self):
        order = np.random.permutation(self.numObjects)
        alpha = max(0.01, 0.05 + 0.02 * np.random.randn())
        return {"order": order, "alpha": alpha}

    def print_individual_order(self, population):
        """Print the permutation of the individuals"""
        for idx, order in enumerate(population):
            print("The order of individual #", idx, "is", order)

    def fitness(self, population):
        """It returns the objects that they are in Knapsack and also the values"""
        knapsack_items_list = []
        value_list = []
        for idx, individual in enumerate(population):
            value = 0
            remainingCapacity = self.capacity

            for ii in individual["order"]:
                if self.weights[ii] <= remainingCapacity:
                    value += self.value[ii]
                    remainingCapacity -= self.weights[ii]
            # print("The remaining capacity for individual #", idx, " is: ", remainingCapacity)
            # print("The objective value for individual #", idx, " is: ", value)
            value_list.append(value)
        return value_list

    def inKnapsack(self, individual):
        """It returns the objects that they are in Knapsack and also the values"""
        knapsack_items = []
        remainingCapacity = self.capacity
        for ii in individual["order"]:
            if self.weights[ii] <= remainingCapacity:
                knapsack_items.append(ii)
                remainingCapacity -= self.weights[ii]
        # if len(population) == 1:
        #    print("The knapsack items for individual #", "are: ", knapsack_items)
        #    print("The remaining capacity for individual #", " is: ", remainingCapacity)

        return knapsack_items

    def initialize(self, lambda_):
        """Initialization of population"""
        print("The number of individuals are:", lambda_)
        individuals = [self.individual() for i in range(lambda_)]

        return individuals

    def mutation(self, individual):
        if np.random.rand() < individual["alpha"]:
            i = np.random.randint(len(individual["order"]))
            j = np.random.randint(len(individual["order"]))
            temp = individual["order"][i]
            individual["order"][i] = individual["order"][j]
            individual["order"][j] = temp
            # print("The new order for individual after mutation is:", order)
        return individual

    def selection(self, population, values):
        """k-tournament selection : select k random individuals and
        find the one with the highest fitness value"""
        k = 5
        indices = np.random.randint(0, len(population), 5)
        selected = list(population[i] for i in indices)
        values_from_selected = list(values[i] for i in indices)
        max_value_idx = values_from_selected.index(max(values_from_selected))

        # print("From the k-tournament selection, we selected: ", selected[max_value_idx], "with fitness value:", values_from_selected[max_value_idx])
        return [selected[max_value_idx], values_from_selected[max_value_idx]]

    def recombination(self, p1, p2):
        s1 = self.inKnapsack(p1)
        s2 = self.inKnapsack(p2)
        """Copy intersection to offspring"""
        offspring = list(set(s1) & set(s2))
        # print("The intersection is:", offspring)
        """Copy in symmetric difference with p% probability"""
        for ii in set(s1).symmetric_difference(set(s2)):
            if np.random.rand() <= 0.5:
                offspring.append(ii)
        # print("Offspring order is:", offspring)

        order = np.array(range(0, self.numObjects))
        i = 0
        for obj in offspring:
            order[i] = obj
            i += 1
        rem = list(set(list(range(0, self.numObjects))).difference(set(offspring)))
        # print("The remaining items are:", rem)
        for obj in rem:
            order[i] = obj
            i += 1
        # print("The order before the permutation of offspring and remaining values are:", order)
        # Randomly permute the elements of offspring
        order[:len(offspring)] = order[np.random.permutation(len(offspring))]
        order[len(offspring):] = order[np.random.permutation(range(len(offspring), self.numObjects))]
        # print("The final order after the permutation is :", order)
        beta = 2 * np.random.rand() - 0.5
        alpha = p1["alpha"] + beta * (p2["alpha"] - p1["alpha"])
        return {"order": order, "alpha": alpha}

    def elimination(self, offspring, population, lambda_individuals):
        combined = population
        for x in offspring:
            combined.append(x)
        # print(len(combined))
        new_combined = list(combined[i] for i in np.argsort(self.fitness(combined)))[::-1]
        return new_combined[0:lambda_individuals]

    def evolutionaryAlgorithm(self):
        lambda_individuals = 100
        mean = 100
        population = self.initialize(lambda_individuals)
        values_population = self.fitness(population)
        iterations = 100
        offspring = [None] * mean
        for ii in range(iterations):
            """Recombination Step"""
            for jj in range(mean):
                p1, _ = self.selection(population, values_population)
                p2, _ = self.selection(population, values_population)
                offspring[jj] = self.recombination(p1, p2)
                offspring[jj] = self.mutation(offspring[jj])

            for kk in range(lambda_individuals):
                population[kk] = self.mutation(population[kk])

            """Elimination Step"""
            population = self.elimination(population, offspring, lambda_individuals)
            fitnesses = self.fitness(population)
            kns = self.inKnapsack(population[fitnesses.index(max(fitnesses))])
            print(ii, "Mean fitness =  ", statistics.mean(fitnesses), "Best fitness:", max(fitnesses), "Knapsack:", kns)


def heuristic_individual(order):
    order = order
    alpha = max(0.01, 0.05 + 0.02 * np.random.randn())
    return {"order": order, "alpha": alpha}


if __name__ == "__main__":
    kp = Knapsack_problem(30)
    heurOrder = np.argsort((kp.value / kp.weights))[::-1]
    heurBest = heuristic_individual(heurOrder)
    kp.evolutionaryAlgorithm()
    print("Heuristic objective value = ", kp.fitness([heurBest]))

from sklearn.model_selection import cross_val_score
import numpy as np
import random
import lightgbm as lgb
from config import *


class FeatureSelector:

    def __init__(self, train,y_train):
        self.train = train
        self.y_train = y_train

    def feature_selection_based_on_genetic_algo(self, train,y_train):
        print ('Number of features : %d' % train.shape[1])

        all_features = train.columns

        """ number of iterations is the only stopping criteria """
        #TODO: implement early stopping based on not getting better score after some iterations

        solutions_dict = dict()
        num_of_features = train.shape[1]

        print(num_of_features)

        """in each loop we get half of the solutions as parents for the next generation"""

        num_of_selected_solutions = NUM_SOLUTIONS/2


        """Creates randomly the initial population"""

        for i in range(NUM_SOLUTIONS):
            prop = random.uniform(0.05,1)
            solutions_dict[i] = np.random.choice([0, 1], size=num_of_features, p=[prop, 1-prop])

        """Scoring Each Solution : use as Fit Functions the mean value of a 5-Fold
        cross validation  based on Lightboost Algo"""

        for i in range(NUM_OF_ITER):
            print('number od iter-------' + str(i))
            solutions_scores = np.zeros(NUM_SOLUTIONS)

            for k in range(NUM_SOLUTIONS):
                cols = []
                solution = solutions_dict[k]
                for j in range(num_of_features):
                    if solution[j] == 1:
                        cols.append(all_features[j])

                et = lgb.LGBMRegressor(boosting_type='gbdt', subsample=1, subsample_freq=1,n_estimators=50, max_depth=3,
                                       scale_pos_weight=5,is_unbalance = True, seed=0, nthread=-1, silent=True,
                                       reg_alpha = 0, reg_lambda=1)

                results = cross_val_score(et, train[cols].values, y_train, cv=5, scoring='r2')
                solutions_scores[k]=results.mean()
            print(solutions_scores.max())

            """ Selection: selecting half solution with the biggest score """

            ind = np.argpartition(solutions_scores, -num_of_selected_solutions)[-num_of_selected_solutions:]
            solutions_dict_new = solutions_dict
            for a in range(num_of_selected_solutions):
                solutions_dict[a] = solutions_dict_new[ind[a]]

            """Crossover : Only one crossing - point implemented here"""

            for b in range(num_of_selected_solutions, NUM_SOLUTIONS,2):
                proportion = random.randint(0, num_of_features - 1)
                solutions_dict[b] = np.concatenate((solutions_dict[b-num_of_selected_solutions][:proportion],
                                                    solutions_dict[b-num_of_selected_solutions+1][proportion:]), axis=0)
                solutions_dict[b+1] = np.concatenate((solutions_dict[b-num_of_selected_solutions+1][:proportion],
                                                      solutions_dict[b-num_of_selected_solutions][proportion:]), axis=0)

            """Mutation"""

            for c in range(NUM_SOLUTIONS):
                mutation_random = random.uniform(0,1)
                mutation_index = random.randint(0, num_of_features - 1)
                if mutation_random < MUTATION_THRESHOLD:
                    solutions_dict[c][mutation_index] = 1-solutions_dict[c][mutation_index]

        """sorts and gets the best solutions"""
        ind = np.argpartition(solutions_scores, -1)[-1:]
        solutions_dict_new = solutions_dict
        solution = dict()
        for a in range(1):
            solutions_dict[a] = solutions_dict_new[ind[a]]

        """gets the columns in the best solution"""
        for k in range(1):
            cols = []
            solution = solutions_dict[k]
            for j in range(num_of_features):
                if solution[j] == 1:
                    cols.append(all_features[j])

        print(cols)
        return cols


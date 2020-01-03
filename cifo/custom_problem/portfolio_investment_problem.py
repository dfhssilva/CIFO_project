from copy import deepcopy
from random import choice, randint
import numpy as np
from pandas import read_excel

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

pip_encoding_rule = {
    "Size": -1,  # It must be defined by the size of DV (Number of products)
    "Is ordered": False,
    "Can repeat": True,
    "Data": [0, 0],
    "Data Type": ""
}

closing_prices = read_excel(r".\data\sp_12_weeks.xlsx")  # importing data example


def get_dv_pip(prices):
    est_ret = np.log(prices.values[1:] / prices.values[:-1])  # returns for each period in column
    exp_ret = np.sum(est_ret, axis=0)  # expected return over time interval
    cov_ret = np.cov(est_ret, rowvar=False)  # estimated covariance matrix over time interval
    return exp_ret, cov_ret


exp, cov = get_dv_pip(closing_prices)
# pip_decision_variables_example = {
#     "Closing_Prices" : closing_prices,  # The closing prices for each period
# }

# Just like in knapsack, each portfolio (bag) is composed by stocks (items) which have an expected return (value) and
# a risk (weight). The main difference is that the stocks (items) are not independent and we need to take into account
# the covariance between them.
pip_decision_variables_example = {
    "Expected_Returns" : exp,  # The closing prices for each period
    "Covariance_Returns": cov
}

pip_constraints_example = {
    "Risk-Tolerance": 1,
    "Budget": 100000
}

# -------------------------------------------------------------------------------------------------
# PIP - Portfolio Investment Problem 
# -------------------------------------------------------------------------------------------------
class PortfolioInvestmentProblem(ProblemTemplate):
    """
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints, encoding_rule=pip_encoding_rule):
        """
        """
        # optimize the access to the decision variables
        # self._prices = np.array([])
        # if "Closing_Prices" in decision_variables:
        #     self._prices = decision_variables["Closing_Prices"]
        #
        # encoding_rule["Size"] = self._prices.shape[1]  # the encoding will be a string with as many elements as stocks
        # encoding_rule["Data"] = decision_variables["Closing_Prices"]
        self._exp_returns = np.array([])
        if "Expected_Returns" in decision_variables:
            self._exp_returns = decision_variables["Expected_Returns"]

        self._cov_returns = np.array([])
        if "Covariance_Returns" in decision_variables:
            self._cov_returns = decision_variables["Covariance_Returns"]

        self._risk_tol = 0  # "Tolerance"
        if "Tolerance" in constraints:
            self._risk_tol = constraints["Tolerance"]

        self._budget = 0  # "Budget"
        if "Budget" in constraints:
            self._budget = constraints["Budget"]

        # the encoding will be a string with as many elements as stocks
        encoding_rule["Size"] = self._exp_returns.shape[0]

        # encoding data doesn't need to be defined (just like in knapsack)

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables=decision_variables,
            constraints=constraints,
            encoding_rule=encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "Portfolio Investment Problem"

        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Maximization

    # Build Solution for Knapsack Problem
    # ----------------------------------------------------------------------------------------------
    def build_solution(self, method='Random'):
        """
        """
        if method == 'Random':  # shuffles the list and then instantiates it as a Linear Solution
            solution_representation = np.random.random(self._encoding.encoding_size)

            solution = LinearSolution(
                representation = solution_representation,
                encoding_rule = self._encoding_rule
            )

            return solution
        else:
            print("Not yet implemented")
            return None

    # Solution Admissibility Function - is_admissible()
    # ----------------------------------------------------------------------------------------------
    def is_admissible(self, solution):  # << use this signature in the sub classes, the meta-heuristic
        """
        """
        weights = solution.representation
        # EXPECTED RETURN AND VARIANCE AS ATTRIBUTES OF SOLUTION? CREATE CHILD CLASS OF LINEARSOLUTION TO HOLD THIS ATTRIBUTES?
        # EVERY SINGLE TIME THIS FUNCTION IS CALLED WE NEED TO CALCULATE THESE
        portfolio_exp_ret = weights @ self._exp_returns  # Portfolio Expected Return formula
        portfolio_var_ret = weights.T @ self._cov_returns @ weights  # Portfolio Variance formula
        risk_free_ret = 0 # Best available rate of return of a risk-free security - FIND VALUE IN DATE
        portfolio_sharpe_ratio = (portfolio_exp_ret - risk_free_ret) / portfolio_var_ret

        if portfolio_sharpe_ratio < self._risk_tol:
            return False
        else:
            return True


    # Evaluate_solution()
    # -------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback=None):  # << This method does not need to be extended, it already
        # automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        """
        # est_ret = np.log(self._prices[1:] / self._prices[:-1])  # returns for each period in column
        # exp_ret = np.sum(est_ret, axis=0)  # expected return over time interval
        # cov_ret = np.cov(est_ret, rowvar=False)  # estimated covariance matrix over time interval

        #The objective is to maximize the return of the portfolio
        weights = solution.representation
        portfolio_exp_ret = weights @ self._exp_returns  # Portfolio Expected Return formula
        # portfolio_var_ret = weights.T @ self._cov_returns @ weights  # Portfolio Variance formula

        solution._fitness = portfolio_exp_ret
        solution._is_fitness_calculated = True

        return solution

    # -------------------------------------------------------------------------------------------------


# OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def pip_bitflip_get_neighbors(solution, problem, neighborhood_size=0):
    pass
